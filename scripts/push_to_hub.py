#!/usr/bin/env python3
"""Script to push trained models to HuggingFace Hub.

This script loads a trained E2D2 model from a checkpoint directory
and pushes it to the HuggingFace Hub for sharing and deployment.

Example:
    Push model to HuggingFace Hub:
    ```
    python scripts/push_to_hub.py \\
        --ckpt_dir path/to/checkpoint \\
        --repo_id username/model-name \\
        --use_ema
    ```

    Save model locally for testing:
    ```
    python scripts/push_to_hub.py \\
        --ckpt_dir path/to/checkpoint \\
        --local_path ./saved_model \\
        --local
    ```
"""

import argparse
from pathlib import Path

import hydra
import yaml
from omegaconf import OmegaConf
from transformers import AutoModelForMaskedLM

from scripts.utils import load_model_from_ckpt_dir_path
from src.logging_config import get_logger
from src.utils import save_pretrained_or_push_to_hub

logger = get_logger(__name__)


def load_model_and_tokenizer(
    ckpt_dir: Path,
    ckpt_file: str = "best-rank0.pt",
    use_ema: bool = True,
):
    """Load model and tokenizer from checkpoint directory.

    Args:
        ckpt_dir: Path to checkpoint directory containing config.yaml and
            checkpoint file.
        ckpt_file: Name of checkpoint file. Defaults to "best-rank0.pt".
        use_ema: Whether to load EMA weights. Defaults to True.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        FileNotFoundError: If checkpoint directory or config file doesn't exist.
    """
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    config_path = ckpt_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from {config_path}")
    with open(config_path, "rb") as f:
        config = yaml.safe_load(f)
    config = OmegaConf.create(config)

    logger.info("Initializing tokenizer")
    tokenizer = hydra.utils.instantiate(config.tokenizer)

    logger.info(f"Loading model from {ckpt_dir / ckpt_file} (use_ema={use_ema})")
    model = load_model_from_ckpt_dir_path(
        path_to_ckpt_dir=str(ckpt_dir),
        ckpt_file=ckpt_file,
        load_ema_weights=use_ema,
    )

    return model, tokenizer


def verify_model(repo_id: str) -> None:
    """Verify model can be loaded from HuggingFace Hub or local path.

    Args:
        repo_id: HuggingFace repo ID or local path to saved model.

    Raises:
        RuntimeError: If model cannot be loaded.
    """
    logger.info(f"Verifying model can be loaded from {repo_id}")
    try:
        test_model = AutoModelForMaskedLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
        )
        logger.info(f"Successfully loaded model: {type(test_model).__name__}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {repo_id}: {e}") from e


def main(args: argparse.Namespace) -> None:
    """Main function to push model to HuggingFace Hub.

    Args:
        args: Command-line arguments.
    """
    ckpt_dir = Path(args.ckpt_dir).resolve()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        ckpt_dir=ckpt_dir,
        ckpt_file=args.ckpt_file,
        use_ema=args.use_ema,
    )

    # Determine destination
    if args.local:
        if args.local_path is None:
            raise ValueError("--local_path must be specified when --local is set")
        repo_id = args.local_path
        logger.info(f"Saving model locally to {repo_id}")
    else:
        if args.repo_id is None:
            raise ValueError("--repo_id must be specified when pushing to Hub")
        repo_id = args.repo_id
        logger.info(f"Pushing model to HuggingFace Hub: {repo_id}")

    # Save or push model
    save_pretrained_or_push_to_hub(
        model=model,
        tokenizer=tokenizer,
        repo_id=repo_id,
        private=args.private,
        local=args.local,
    )

    # Verify model can be loaded
    if args.verify:
        verify_model(repo_id)

    logger.info("Done!")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Push trained E2D2 model to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory containing config.yaml and checkpoint file",
    )

    # Model loading arguments
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="best-rank0.pt",
        help="Name of checkpoint file to load",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Load EMA weights instead of regular weights",
    )

    # Destination arguments
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--local_path",
        type=str,
        default=None,
        help="Local directory path for saving model (used with --local)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Save model locally instead of pushing to Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make HuggingFace repository private (only used when pushing to Hub)",
    )

    # Verification
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify model can be loaded after saving/pushing",
    )
    parser.add_argument(
        "--no_verify",
        dest="verify",
        action="store_false",
        help="Skip verification step",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
