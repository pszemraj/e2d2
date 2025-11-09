"""Type definitions and Pydantic models for the E2D2 project.

This module provides validated configuration classes and type aliases
to ensure type safety and runtime validation throughout the codebase.
"""

from enum import Enum
from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelType(str, Enum):
    """Supported model types."""

    AR = "ar"  # Autoregressive
    D3PM = "d3pm"  # Discrete Denoising Diffusion Probabilistic Model
    MDLM = "mdlm"  # Masked Diffusion Language Model
    BD3LM = "bd3lm"  # Block Diffusion Language Model
    E2D2 = "e2d2"  # Encoder-Decoder Diffusion Language Model


class NoiseScheduleType(str, Enum):
    """Supported noise schedule types."""

    COSINE = "cosine"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"


class SamplingMethod(str, Enum):
    """Sampling methods for generation."""

    GREEDY = "greedy"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TEMPERATURE = "temperature"
    GUMBEL = "gumbel"


class TaskType(str, Enum):
    """Task types for evaluation and training."""

    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    LANGUAGE_MODELING = "language_modeling"


class BaseConfig(BaseModel):
    """Base configuration with common settings."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        use_enum_values=False,
    )


class NoiseScheduleConfig(BaseConfig):
    """Configuration for noise schedules.

    Attributes:
        schedule_type: Type of noise schedule to use.
        eps: Epsilon value for numerical stability.
        num_steps: Number of diffusion steps (if applicable).
    """

    schedule_type: NoiseScheduleType
    eps: float = Field(default=1e-3, gt=0, lt=1)
    num_steps: Optional[int] = Field(default=None, ge=1)

    @field_validator("eps")
    @classmethod
    def validate_eps(cls, v: float) -> float:
        """Ensure epsilon is in reasonable range."""
        if not (1e-10 < v < 1e-1):
            raise ValueError(f"eps must be between 1e-10 and 1e-1, got {v}")
        return v


class GenerationConfig(BaseConfig):
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate.
        min_new_tokens: Minimum number of new tokens to generate.
        num_iterations: Number of denoising iterations (for diffusion models).
        sampling_method: Method to use for sampling.
        temperature: Temperature for sampling (if applicable).
        top_k: Top-k value for top-k sampling.
        top_p: Top-p value for nucleus sampling.
        do_sample: Whether to use sampling (vs greedy decoding).
    """

    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    min_new_tokens: int = Field(default=0, ge=0)
    num_iterations: Optional[int] = Field(default=None, ge=1)
    sampling_method: SamplingMethod = SamplingMethod.GREEDY
    temperature: float = Field(default=1.0, gt=0)
    top_k: Optional[int] = Field(default=None, ge=1)
    top_p: Optional[float] = Field(default=None, gt=0, lt=1)
    do_sample: bool = False

    @field_validator("min_new_tokens")
    @classmethod
    def validate_min_tokens(cls, v: int, info: dict) -> int:
        """Ensure min_new_tokens <= max_new_tokens."""
        if "max_new_tokens" in info.data and v > info.data["max_new_tokens"]:
            raise ValueError(
                f"min_new_tokens ({v}) must be <= max_new_tokens "
                f"({info.data['max_new_tokens']})"
            )
        return v


class DatasetConfig(BaseConfig):
    """Configuration for dataset loading and preprocessing.

    Attributes:
        name: Name of the dataset.
        task_type: Type of task (translation, summarization, etc.).
        split: Dataset split to use (train, validation, test).
        max_length: Maximum sequence length.
        batch_size: Batch size for data loading.
        num_workers: Number of workers for data loading.
        source_lang: Source language (for translation).
        target_lang: Target language (for translation).
    """

    name: str
    task_type: TaskType
    split: Literal["train", "validation", "test"] = "train"
    max_length: int = Field(default=512, ge=1, le=8192)
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None

    @field_validator("source_lang", "target_lang")
    @classmethod
    def validate_translation_langs(cls, v: Optional[str], info: dict) -> Optional[str]:
        """Ensure translation tasks have source and target languages."""
        if (
            info.data.get("task_type") == TaskType.TRANSLATION
            and v is None
            and info.field_name in ["source_lang", "target_lang"]
        ):
            raise ValueError(
                f"{info.field_name} is required for translation tasks"
            )
        return v


class TrainingConfig(BaseConfig):
    """Configuration for training.

    Attributes:
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        max_steps: Maximum number of training steps.
        warmup_steps: Number of warmup steps for learning rate schedule.
        gradient_clip_norm: Maximum gradient norm for clipping.
        seed: Random seed for reproducibility.
        device: Device to use for training.
        mixed_precision: Whether to use mixed precision training.
    """

    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    max_steps: int = Field(default=100000, ge=1)
    warmup_steps: int = Field(default=1000, ge=0)
    gradient_clip_norm: Optional[float] = Field(default=1.0, gt=0)
    seed: int = Field(default=42, ge=0)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = False

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure device is valid."""
        if v not in ["cpu", "cuda", "mps"]:
            if not v.startswith("cuda:"):
                raise ValueError(
                    f"Invalid device: {v}. Must be 'cpu', 'cuda', 'mps', or 'cuda:N'"
                )
        return v
