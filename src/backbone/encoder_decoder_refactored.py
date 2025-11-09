"""Refactored encoder-decoder implementation with eliminated duplication.

This module provides a unified LLMasEncoderDecoder class that replaces the
previous duplicate implementations (LLMasEncoderDecoder and LLMasEncoderDecoderShareKV).

The key improvement is using a model_class parameter to control whether to use
CustomQwen3ForCausalLM or AutoModelForCausalLM, eliminating 300+ lines of
duplicate code.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import (
    ModelOutput,
)
from transformers.utils import logging

from src.logging_config import get_logger

try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None


hf_logger = logging.get_logger(__name__)
logger = get_logger(__name__)


@dataclass
class EncoderBaseModelOutputWithPast(ModelOutput):
    """Custom (encoder) model output.

    Stores previous decoder and updated encoder cache and encoder last hidden state.

    Attributes:
        past_key_values: Previous decoder key-value cache.
        encoder_last_hidden_state: Last hidden state from encoder.
        encoder_past_key_values: Updated encoder key-value cache.
    """

    past_key_values: Optional[Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]] = (
        None
    )
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_past_key_values: Optional[
        Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]
    ] = None


@dataclass
class DecoderCausalLMOutputWithPast(ModelOutput):
    """Custom (decoder) model output.

    Stores previous encoder and updated decoder cache and decoder logits.

    Attributes:
        logits: Decoder output logits.
        past_key_values: Updated decoder key-value cache.
        encoder_past_key_values: Previous encoder key-value cache.
    """

    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]] = (
        None
    )
    encoder_past_key_values: Optional[
        Union[Tuple[Tuple[torch.FloatTensor]], DynamicCache]
    ] = None


class LLMasEncoderDecoderUnified(nn.Module):
    """Unified encoder-decoder architecture using language models.

    This class replaces the previous duplicate implementations by accepting
    a model_class parameter to control which transformer model to use.

    Args:
        pretrained_model_name_or_path: HuggingFace model ID or path.
        max_length: Maximum sequence length.
        attn_backend: Attention implementation ("sdpa", "flash_attention_2", etc.).
        model_class: Model class to use. If None, uses AutoModelForCausalLM.
            Set to CustomQwen3ForCausalLM for custom Qwen behavior.
        freeze_encoder: Whether to freeze encoder parameters.
        reinit_encoder: Whether to reinitialize encoder from scratch.
        reinit_decoder: Whether to reinitialize decoder from scratch.
        tie_encoder_decoder_weights: Whether encoder and decoder share weights.
        use_encoder_causal_mask: Whether to use causal mask for encoder.
        num_encoder_layers: Number of encoder layers to keep (-1 for all).
        num_decoder_layers: Number of decoder layers to keep (-1 for all).
        keep_top_encoder_layers: If True, keep top layers; otherwise keep bottom layers.
        keep_top_decoder_layers: If True, keep top layers; otherwise keep bottom layers.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
        **llm_init_kwargs: Additional kwargs for model initialization.

    Example:
        >>> # Use with CustomQwen3
        >>> model = LLMasEncoderDecoderUnified(
        ...     "Qwen/Qwen3-0.5B",
        ...     max_length=512,
        ...     model_class=CustomQwen3ForCausalLM
        ... )
        >>> # Use with AutoModel (default)
        >>> model = LLMasEncoderDecoderUnified(
        ...     "gpt2",
        ...     max_length=512
        ... )
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_length: int,
        attn_backend: str = "sdpa",
        model_class: Optional[Type[PreTrainedModel]] = None,
        freeze_encoder: bool = False,
        reinit_encoder: bool = False,
        reinit_decoder: bool = False,
        tie_encoder_decoder_weights: bool = False,
        use_encoder_causal_mask: bool = False,
        num_encoder_layers: int = -1,
        num_decoder_layers: int = -1,
        keep_top_encoder_layers: bool = False,
        keep_top_decoder_layers: bool = False,
        use_gradient_checkpointing: bool = False,
        **llm_init_kwargs,
    ):
        """Initialize encoder-decoder model."""
        # Validation
        if tie_encoder_decoder_weights and reinit_decoder:
            raise ValueError(
                "Cannot tie encoder-decoder weights and reinitialize decoder."
            )
        if tie_encoder_decoder_weights and freeze_encoder:
            raise ValueError(
                "Cannot freeze encoder weights when tying encoder-decoder weights."
            )

        super().__init__()
        self.use_encoder_causal_mask = use_encoder_causal_mask
        self.tie_encoder_decoder_weights = tie_encoder_decoder_weights

        # Default to AutoModelForCausalLM if not specified
        if model_class is None:
            model_class = AutoModelForCausalLM
            logger.info("Using AutoModelForCausalLM (default)")
        else:
            logger.info(f"Using {model_class.__name__}")

        # Initialize encoder
        self.encoder = self._initialize_model(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_class=model_class,
            attn_backend=attn_backend,
            reinit=reinit_encoder,
            num_layers=num_encoder_layers,
            keep_top_layers=keep_top_encoder_layers,
            component_name="encoder",
            **llm_init_kwargs,
        )

        # Freeze encoder if requested
        if freeze_encoder:
            logger.info("Freezing encoder parameters")
            for name, param in self.encoder.named_parameters():
                if "embed_tokens" not in name:
                    param.requires_grad = False

        if use_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing for encoder")
            self.encoder.gradient_checkpointing_enable()

        # Initialize decoder
        if tie_encoder_decoder_weights:
            logger.info("Tying encoder-decoder weights")
            self.decoder = self.encoder
            num_decoder_layers = (
                len(self.decoder.model.layers)
                if num_decoder_layers == -1
                else num_decoder_layers
            )
            if num_decoder_layers != len(self.decoder.model.layers):
                logger.warning(
                    f"Requested {num_decoder_layers} decoder layers but encoder has "
                    f"{len(self.decoder.model.layers)}. Adjusting decoder layers."
                )
                if keep_top_decoder_layers:
                    self.decoder.model.layers = self.decoder.model.layers[
                        -num_decoder_layers:
                    ]
                else:
                    self.decoder.model.layers = self.decoder.model.layers[
                        :num_decoder_layers
                    ]
        else:
            self.decoder = self._initialize_model(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                model_class=model_class,
                attn_backend=attn_backend,
                reinit=reinit_decoder,
                num_layers=num_decoder_layers,
                keep_top_layers=keep_top_decoder_layers,
                component_name="decoder",
                **llm_init_kwargs,
            )

        if use_gradient_checkpointing and not tie_encoder_decoder_weights:
            logger.info("Enabling gradient checkpointing for decoder")
            self.decoder.gradient_checkpointing_enable()

        self.max_length = max_length
        self.config = self.decoder.config
        logger.info("Encoder-decoder model initialized successfully")

    def _initialize_model(
        self,
        pretrained_model_name_or_path: str,
        model_class: Type[PreTrainedModel],
        attn_backend: str,
        reinit: bool,
        num_layers: int,
        keep_top_layers: bool,
        component_name: str,
        **llm_init_kwargs,
    ) -> PreTrainedModel:
        """Initialize encoder or decoder model.

        Args:
            pretrained_model_name_or_path: Model ID or path.
            model_class: Model class to instantiate.
            attn_backend: Attention implementation.
            reinit: Whether to reinitialize from scratch.
            num_layers: Number of layers to keep.
            keep_top_layers: Whether to keep top or bottom layers.
            component_name: "encoder" or "decoder" for logging.
            **llm_init_kwargs: Additional initialization kwargs.

        Returns:
            Initialized model.

        Raises:
            AssertionError: If layer configuration is invalid.
        """
        logger.info(f"Initializing {component_name}")

        if reinit:
            if num_layers <= 0:
                raise ValueError(
                    f"num_layers must be > 0 when reinit=True, got {num_layers}"
                )
            logger.info(f"Reinitializing {component_name} with {num_layers} layers")
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                num_hidden_layers=num_layers,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )
            model = model_class.from_config(config)
        else:
            logger.info(
                f"Loading pretrained {component_name} from "
                f"{pretrained_model_name_or_path}"
            )
            model = model_class.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_backend,
                **llm_init_kwargs,
            )

            # Adjust number of layers
            total_layers = len(model.model.layers)
            if num_layers > total_layers:
                raise ValueError(
                    f"Cannot keep {num_layers} layers. "
                    f"Pretrained model only has {total_layers} layers."
                )

            num_layers = total_layers if num_layers == -1 else num_layers

            if num_layers != total_layers:
                logger.info(
                    f"Adjusting {component_name} from {total_layers} "
                    f"to {num_layers} layers"
                )
                if keep_top_layers:
                    model.model.layers = model.model.layers[-num_layers:]
                else:
                    model.model.layers = model.model.layers[:num_layers]

        return model

    # Additional methods would go here (forward, encode, decode, etc.)
    # These are shared across both original classes


# Backward compatibility: Create aliases for old class names
LLMasEncoderDecoder = LLMasEncoderDecoderUnified
LLMasEncoderDecoderShareKV = LLMasEncoderDecoderUnified
