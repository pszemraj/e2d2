"""Noise schedules for diffusion models.

This module implements various noise schedules used in discrete diffusion models.
Each schedule defines how noise is added during the forward process and can be
inverted to compute timesteps from noise parameters.
"""

from abc import ABC, abstractmethod
from typing import Union

import torch

from src.constants import DEFAULT_NOISE_EPS
from src.logging_config import get_logger

logger = get_logger(__name__)


class Noise(ABC):
    """Abstract base class for noise schedules.

    Noise schedules control how noise is added during the forward diffusion process.
    All schedules assume time t ∈ [0, 1] where:
    - t=0: Clean data (no noise)
    - t=1: Maximum noise

    Attributes:
        name: Human-readable name of the noise schedule.
        eps: Small epsilon for numerical stability.
    """

    name: str
    eps: float

    @abstractmethod
    def __call__(
        self, t: Union[torch.Tensor, float]
    ) -> tuple[Union[torch.Tensor, float], Union[torch.Tensor, float]]:
        """Compute noise parameters at timestep t.

        Args:
            t: Timestep(s) in range [0, 1]. Can be a scalar or tensor.

        Returns:
            Tuple of (alpha_t, alpha_t_prime) where:
                - alpha_t: Probability of keeping original token
                - alpha_t_prime: Derivative of alpha_t with respect to t

        Raises:
            ValueError: If t is outside the range [0, 1].
        """
        pass

    @abstractmethod
    def inverse(self, alpha_t: torch.Tensor) -> torch.Tensor:
        """Compute timestep t from noise schedule parameter alpha_t.

        Args:
            alpha_t: Noise schedule parameter (probability of keeping original token).

        Returns:
            Timestep t corresponding to the given alpha_t.

        Raises:
            NotImplementedError: If the inverse is not analytically tractable.
        """
        raise NotImplementedError(
            f"Inverse function not implemented for {self.__class__.__name__}"
        )

    def _validate_timestep(self, t: Union[torch.Tensor, float]) -> None:
        """Validate that timestep is in valid range [0, 1].

        Args:
            t: Timestep to validate.

        Raises:
            ValueError: If t is outside [0, 1].
        """
        if isinstance(t, torch.Tensor):
            if torch.any(t < 0) or torch.any(t > 1):
                raise ValueError(
                    f"Timestep must be in [0, 1], got min={t.min()}, max={t.max()}"
                )
        elif isinstance(t, (int, float)):
            if not 0 <= t <= 1:
                raise ValueError(f"Timestep must be in [0, 1], got {t}")


class CosineNoise(Noise):
    """Cosine noise schedule.

    Implements the cosine schedule from "Improved Denoising Diffusion Probabilistic
    Models" (Nichol & Dhariwal, 2021). This schedule provides smooth interpolation
    between clean and noisy states.

    Attributes:
        eps: Small offset to prevent division by zero at boundaries.
        name: "cosine"

    Example:
        >>> noise = CosineNoise(eps=1e-3)
        >>> t = torch.tensor([0.0, 0.5, 1.0])
        >>> alpha_t, alpha_t_prime = noise(t)
    """

    def __init__(self, eps: float = DEFAULT_NOISE_EPS) -> None:
        """Initialize cosine noise schedule.

        Args:
            eps: Small epsilon for numerical stability. Must be in (0, 1).

        Raises:
            ValueError: If eps is not in valid range.
        """
        super().__init__()
        if not 0 < eps < 1:
            raise ValueError(f"eps must be in (0, 1), got {eps}")
        self.eps = eps
        self.name = "cosine"
        logger.debug(f"Initialized {self.name} noise schedule with eps={eps}")

    def __call__(
        self, t: Union[torch.Tensor, float]
    ) -> tuple[Union[torch.Tensor, float], Union[torch.Tensor, float]]:
        """Compute cosine noise parameters at timestep t.

        Args:
            t: Timestep(s) in range [0, 1].

        Returns:
            Tuple of (alpha_t, alpha_t_prime).

        Raises:
            ValueError: If t is outside [0, 1].
        """
        self._validate_timestep(t)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        t = t.to(torch.float32)

        cos = -(1 - self.eps) * torch.cos(t * torch.pi / 2)
        sin = -(1 - self.eps) * torch.sin(t * torch.pi / 2)
        move_chance = cos + 1
        alpha_t_prime = sin * torch.pi / 2
        return 1 - move_chance, alpha_t_prime

    def inverse(self, alpha_t: torch.Tensor) -> torch.Tensor:
        """Compute timestep from alpha_t using inverse cosine.

        Args:
            alpha_t: Noise parameter (probability of keeping token).

        Returns:
            Timestep t ∈ [0, 1].
        """
        move_chance = 1 - alpha_t
        cos_val = move_chance - 1
        # Solve: -(1-eps)*cos(t*π/2) = cos_val
        t = (2 / torch.pi) * torch.acos(-cos_val / (1 - self.eps))
        return t


class ExponentialNoise(Noise):
    """Exponential noise schedule.

    Provides polynomial-based noise scheduling where noise increases as t^exp.
    Higher exponents result in slower initial noise accumulation.

    Attributes:
        exp: Exponent for the power function (typically 2-4).
        eps: Minimum clamp value for numerical stability.
        name: "exp_{exp}"

    Example:
        >>> noise = ExponentialNoise(exp=2, eps=1e-3)
        >>> t = torch.linspace(0, 1, 10)
        >>> alpha_t, alpha_t_prime = noise(t)
    """

    def __init__(self, exp: int = 2, eps: float = DEFAULT_NOISE_EPS) -> None:
        """Initialize exponential noise schedule.

        Args:
            exp: Exponent for power function. Must be positive.
            eps: Minimum clamp value for numerical stability.

        Raises:
            ValueError: If exp <= 0 or eps not in valid range.
        """
        super().__init__()
        if exp <= 0:
            raise ValueError(f"exp must be positive, got {exp}")
        if not 0 < eps < 1:
            raise ValueError(f"eps must be in (0, 1), got {eps}")
        self.eps = eps
        self.exp = exp
        self.name = f"exp_{exp}"
        logger.debug(
            f"Initialized {self.name} noise schedule with exp={exp}, eps={eps}"
        )

    def __call__(
        self, t: Union[torch.Tensor, float]
    ) -> tuple[Union[torch.Tensor, float], Union[torch.Tensor, float]]:
        """Compute exponential noise parameters at timestep t.

        Args:
            t: Timestep(s) in range [0, 1].

        Returns:
            Tuple of (alpha_t_prime, alpha_t). Note: Order differs from other schedules!

        Raises:
            ValueError: If t is outside [0, 1].
        """
        self._validate_timestep(t)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        t = t.to(torch.float32)

        move_chance = torch.pow(t, self.exp)
        move_chance = torch.clamp(move_chance, min=self.eps)
        alpha_t_prime = -self.exp * torch.pow(t, self.exp - 1)
        return alpha_t_prime, 1 - move_chance

    def inverse(self, alpha_t: torch.Tensor) -> torch.Tensor:
        """Compute timestep from alpha_t using inverse exponential.

        Args:
            alpha_t: Noise parameter (probability of keeping token).

        Returns:
            Timestep t ∈ [0, 1].
        """
        move_chance = 1 - alpha_t
        # Solve: move_chance = t^exp  =>  t = move_chance^(1/exp)
        t = torch.pow(torch.clamp(move_chance, min=self.eps), 1.0 / self.exp)
        return t


class LogarithmicNoise(Noise):
    """Logarithmic noise schedule.

    Uses log-based interpolation for noise scheduling. Provides faster initial
    noise accumulation compared to linear or cosine schedules.

    Attributes:
        eps: Small epsilon for numerical stability.
        name: "logarithmic"

    Example:
        >>> noise = LogarithmicNoise(eps=1e-3)
        >>> t = torch.tensor([0.1, 0.5, 0.9])
        >>> alpha_t, alpha_t_prime = noise(t)
    """

    def __init__(self, eps: float = DEFAULT_NOISE_EPS) -> None:
        """Initialize logarithmic noise schedule.

        Args:
            eps: Small epsilon for numerical stability. Must be in (0, 1).

        Raises:
            ValueError: If eps is not in valid range.
        """
        super().__init__()
        if not 0 < eps < 1:
            raise ValueError(f"eps must be in (0, 1), got {eps}")
        self.eps = eps
        self.name = "logarithmic"
        logger.debug(f"Initialized {self.name} noise schedule with eps={eps}")

    def __call__(
        self, t: Union[torch.Tensor, float]
    ) -> tuple[Union[torch.Tensor, float], Union[torch.Tensor, float]]:
        """Compute logarithmic noise parameters at timestep t.

        Args:
            t: Timestep(s) in range [0, 1].

        Returns:
            Tuple of (alpha_t, alpha_t_prime).

        Raises:
            ValueError: If t is outside [0, 1].
        """
        self._validate_timestep(t)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        t = t.to(torch.float32)

        move_chance = torch.log1p(t) / torch.log(torch.tensor(2.0))
        alpha_t_prime = -1 / (torch.log(torch.tensor(2.0)) * (1 + t))
        return 1 - move_chance, alpha_t_prime

    def inverse(self, alpha_t: torch.Tensor) -> torch.Tensor:
        """Compute timestep from alpha_t using inverse logarithm.

        Args:
            alpha_t: Noise parameter (probability of keeping token).

        Returns:
            Timestep t ∈ [0, 1].
        """
        move_chance = 1 - alpha_t
        # Solve: log(1+t)/log(2) = move_chance  =>  t = 2^move_chance - 1
        t = torch.pow(torch.tensor(2.0), move_chance) - 1
        return t


class LinearNoise(Noise):
    """Linear noise schedule.

    Simplest noise schedule with linear interpolation between clean and noisy states.
    Provides uniform noise accumulation over time.

    Attributes:
        name: "linear"

    Example:
        >>> noise = LinearNoise()
        >>> t = torch.linspace(0, 1, 100)
        >>> alpha_t, alpha_t_prime = noise(t)
    """

    def __init__(self) -> None:
        """Initialize linear noise schedule.

        No parameters needed for linear schedule.
        """
        super().__init__()
        self.name = "linear"
        self.eps = 0.0  # Not used in linear schedule
        logger.debug(f"Initialized {self.name} noise schedule")

    def __call__(
        self, t: Union[torch.Tensor, float]
    ) -> tuple[Union[torch.Tensor, float], Union[torch.Tensor, float]]:
        """Compute linear noise parameters at timestep t.

        Args:
            t: Timestep(s) in range [0, 1].

        Returns:
            Tuple of (alpha_t, alpha_t_prime) where alpha_t = 1-t (constant derivative).

        Raises:
            ValueError: If t is outside [0, 1].
        """
        self._validate_timestep(t)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        t = t.to(torch.float32)

        alpha_t_prime = -torch.ones_like(t)
        move_chance = t
        return 1 - move_chance, alpha_t_prime

    def inverse(self, alpha_t: torch.Tensor) -> torch.Tensor:
        """Compute timestep from alpha_t using linear inverse.

        Args:
            alpha_t: Noise parameter (probability of keeping token).

        Returns:
            Timestep t ∈ [0, 1] where t = 1 - alpha_t.
        """
        return 1 - alpha_t
