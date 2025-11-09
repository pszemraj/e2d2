"""Global constants for the E2D2 project.

This module centralizes all magic numbers and configuration constants
to improve maintainability and avoid scattered hardcoded values.
"""

from typing import Final

# Numerical Constants
EPSILON: Final[float] = 1e-10
"""Small epsilon value for numerical stability in probability computations."""

GUMBEL_EPSILON: Final[float] = 1e-10
"""Epsilon value for Gumbel noise generation."""

NEG_INFINITY: Final[float] = -1e12
"""Large negative value to approximate negative infinity in attention masking."""

MIN_TIMESTEP: Final[float] = 1e-5
"""Minimum timestep value for diffusion process."""

CONFIDENCE_THRESHOLD: Final[float] = 1e6
"""Confidence threshold for sampling decisions."""

# Noise Schedule Constants
DEFAULT_NOISE_EPS: Final[float] = 1e-3
"""Default epsilon for noise schedules."""

# Dataset Prompt Templates
QUESTION_PREFIX: Final[str] = (
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
)
"""Prefix template for question-answering tasks (GSM8K)."""

SUMMARY_PREFIX: Final[str] = "Please summarize the following text: "
"""Prefix template for summarization tasks."""

TRANSLATION_PREFIX_TEMPLATE: Final[str] = (
    "Translate the following text from {source} to {target}: "
)
"""Prefix template for translation tasks. Use .format(source=..., target=...)."""

# Evaluation Constants
THROUGHPUT_SAMPLES: Final[int] = 100
"""Number of samples to use for throughput evaluation."""

THROUGHPUT_WARMUP: Final[int] = 100
"""Number of warmup samples before throughput evaluation."""

# Model Architecture Constants
DEFAULT_PAD_TOKEN: Final[str] = "<|endoftext|>"
"""Default padding token for tokenizers that don't have one."""

# Logging Constants
LOG_FORMAT: Final[str] = (
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
)
"""Standard logging format for the project."""

LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
"""Date format for log messages."""
