"""Unit tests for noise schedule implementations.

Tests cover:
- Initialization and validation
- Forward noise computation
- Inverse computation (timestep recovery)
- Edge cases and error handling
"""

import pytest
import torch

from src.constants import DEFAULT_NOISE_EPS
from src.noise_schedule.noise_schedules import (
    CosineNoise,
    ExponentialNoise,
    LinearNoise,
    LogarithmicNoise,
    Noise,
)


class TestNoiseBase:
    """Base test class for common noise schedule tests."""

    @pytest.fixture
    def noise_schedule(self) -> Noise:
        """Override in subclasses to provide specific noise schedule."""
        raise NotImplementedError

    def test_timestep_validation_tensor(self, noise_schedule: Noise) -> None:
        """Test that invalid timestep tensors raise ValueError."""
        # Test negative timesteps
        with pytest.raises(ValueError, match="Timestep must be in"):
            invalid_t = torch.tensor([-0.1, 0.5, 1.0])
            noise_schedule(invalid_t)

        # Test timesteps > 1
        with pytest.raises(ValueError, match="Timestep must be in"):
            invalid_t = torch.tensor([0.0, 0.5, 1.5])
            noise_schedule(invalid_t)

    def test_timestep_validation_scalar(self, noise_schedule: Noise) -> None:
        """Test that invalid scalar timesteps raise ValueError."""
        with pytest.raises(ValueError, match="Timestep must be in"):
            noise_schedule(-0.1)

        with pytest.raises(ValueError, match="Timestep must be in"):
            noise_schedule(1.5)

    def test_valid_timesteps(self, noise_schedule: Noise) -> None:
        """Test that valid timesteps work correctly."""
        # Test scalar
        alpha_t, alpha_t_prime = noise_schedule(0.5)
        assert isinstance(alpha_t, torch.Tensor)
        assert isinstance(alpha_t_prime, torch.Tensor)

        # Test tensor
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        alpha_t, alpha_t_prime = noise_schedule(t)
        assert isinstance(alpha_t, torch.Tensor)
        assert isinstance(alpha_t_prime, torch.Tensor)
        assert alpha_t.shape == t.shape
        assert alpha_t_prime.shape == t.shape

    def test_boundary_conditions(self, noise_schedule: Noise) -> None:
        """Test behavior at boundary timesteps (0 and 1)."""
        # Handle ExponentialNoise which returns (alpha_t_prime, alpha_t) instead
        result_0 = noise_schedule(torch.tensor(0.0))
        result_1 = noise_schedule(torch.tensor(1.0))

        if isinstance(noise_schedule, ExponentialNoise):
            _, alpha_0 = result_0
            _, alpha_1 = result_1
        else:
            alpha_0, _ = result_0
            alpha_1, _ = result_1

        # At t=0, should have maximum alpha_t (clean data)
        assert alpha_0 > 0.9  # Should be close to 1

        # At t=1, should have minimum alpha_t (maximum noise)
        assert alpha_1 < 0.1  # Should be close to 0

    def test_monotonicity(self, noise_schedule: Noise) -> None:
        """Test that alpha_t decreases monotonically with t."""
        t_values = torch.linspace(0, 1, 100)
        alpha_t_values, _ = noise_schedule(t_values)

        # Check that alpha_t is monotonically decreasing
        for i in range(len(alpha_t_values) - 1):
            assert alpha_t_values[i] >= alpha_t_values[i + 1], (
                f"Non-monotonic at index {i}"
            )


class TestLinearNoise(TestNoiseBase):
    """Tests for LinearNoise schedule."""

    @pytest.fixture
    def noise_schedule(self) -> LinearNoise:
        """Provide LinearNoise instance."""
        return LinearNoise()

    def test_initialization(self) -> None:
        """Test LinearNoise initialization."""
        noise = LinearNoise()
        assert noise.name == "linear"
        assert noise.eps == 0.0

    def test_linear_behavior(self, noise_schedule: LinearNoise) -> None:
        """Test that LinearNoise has constant derivative."""
        t = torch.tensor([0.0, 0.5, 1.0])
        alpha_t, alpha_t_prime = noise_schedule(t)

        # For linear schedule: alpha_t = 1 - t
        expected_alpha_t = 1 - t
        assert torch.allclose(alpha_t, expected_alpha_t, atol=1e-6)

        # Derivative should be -1
        assert torch.allclose(alpha_t_prime, torch.tensor(-1.0), atol=1e-6)

    def test_inverse(self, noise_schedule: LinearNoise) -> None:
        """Test that inverse recovers original timestep."""
        t_original = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        alpha_t, _ = noise_schedule(t_original)
        t_recovered = noise_schedule.inverse(alpha_t)

        assert torch.allclose(t_original, t_recovered, atol=1e-5)


class TestCosineNoise(TestNoiseBase):
    """Tests for CosineNoise schedule."""

    @pytest.fixture
    def noise_schedule(self) -> CosineNoise:
        """Provide CosineNoise instance."""
        return CosineNoise(eps=DEFAULT_NOISE_EPS)

    def test_initialization(self) -> None:
        """Test CosineNoise initialization."""
        noise = CosineNoise(eps=1e-3)
        assert noise.name == "cosine"
        assert noise.eps == 1e-3

    def test_initialization_invalid_eps(self) -> None:
        """Test that invalid eps raises ValueError."""
        with pytest.raises(ValueError, match="eps must be in"):
            CosineNoise(eps=-0.1)

        with pytest.raises(ValueError, match="eps must be in"):
            CosineNoise(eps=1.5)

    def test_smooth_interpolation(self, noise_schedule: CosineNoise) -> None:
        """Test that cosine provides smooth interpolation."""
        t = torch.linspace(0, 1, 100)
        alpha_t, alpha_t_prime = noise_schedule(t)

        # Check smoothness by verifying derivative doesn't have large jumps
        derivative_diff = torch.diff(alpha_t_prime)
        assert torch.all(torch.abs(derivative_diff) < 0.1)

    def test_inverse(self, noise_schedule: CosineNoise) -> None:
        """Test that inverse recovers original timestep."""
        t_original = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        alpha_t, _ = noise_schedule(t_original)
        t_recovered = noise_schedule.inverse(alpha_t)

        assert torch.allclose(t_original, t_recovered, atol=1e-4)


class TestExponentialNoise(TestNoiseBase):
    """Tests for ExponentialNoise schedule."""

    @pytest.fixture
    def noise_schedule(self) -> ExponentialNoise:
        """Provide ExponentialNoise instance."""
        return ExponentialNoise(exp=2, eps=DEFAULT_NOISE_EPS)

    def test_initialization(self) -> None:
        """Test ExponentialNoise initialization."""
        noise = ExponentialNoise(exp=3, eps=1e-3)
        assert noise.name == "exp_3"
        assert noise.eps == 1e-3
        assert noise.exp == 3

    def test_initialization_invalid_exp(self) -> None:
        """Test that invalid exp raises ValueError."""
        with pytest.raises(ValueError, match="exp must be positive"):
            ExponentialNoise(exp=0)

        with pytest.raises(ValueError, match="exp must be positive"):
            ExponentialNoise(exp=-1)

    def test_initialization_invalid_eps(self) -> None:
        """Test that invalid eps raises ValueError."""
        with pytest.raises(ValueError, match="eps must be in"):
            ExponentialNoise(exp=2, eps=-0.1)

    def test_exponential_behavior(self, noise_schedule: ExponentialNoise) -> None:
        """Test exponential power behavior."""
        t = torch.tensor([0.0, 0.5, 1.0])
        alpha_t_prime, alpha_t = noise_schedule(t)  # Note: order is swapped!

        # At t=0.5, with exp=2: move_chance = 0.5^2 = 0.25
        # So alpha_t = 1 - 0.25 = 0.75
        expected_alpha_at_half = 1 - 0.5**2
        actual_alpha_at_half = alpha_t[1]
        assert torch.allclose(
            actual_alpha_at_half, torch.tensor(expected_alpha_at_half), atol=1e-5
        )

    def test_inverse(self, noise_schedule: ExponentialNoise) -> None:
        """Test that inverse recovers original timestep."""
        t_original = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        _, alpha_t = noise_schedule(t_original)  # Note: order is swapped!
        t_recovered = noise_schedule.inverse(alpha_t)

        assert torch.allclose(t_original, t_recovered, atol=1e-4)


class TestLogarithmicNoise(TestNoiseBase):
    """Tests for LogarithmicNoise schedule."""

    @pytest.fixture
    def noise_schedule(self) -> LogarithmicNoise:
        """Provide LogarithmicNoise instance."""
        return LogarithmicNoise(eps=DEFAULT_NOISE_EPS)

    def test_initialization(self) -> None:
        """Test LogarithmicNoise initialization."""
        noise = LogarithmicNoise(eps=1e-3)
        assert noise.name == "logarithmic"
        assert noise.eps == 1e-3

    def test_initialization_invalid_eps(self) -> None:
        """Test that invalid eps raises ValueError."""
        with pytest.raises(ValueError, match="eps must be in"):
            LogarithmicNoise(eps=0)

        with pytest.raises(ValueError, match="eps must be in"):
            LogarithmicNoise(eps=2.0)

    def test_logarithmic_behavior(self, noise_schedule: LogarithmicNoise) -> None:
        """Test logarithmic scaling."""
        t = torch.linspace(0, 1, 100)
        alpha_t, _ = noise_schedule(t)

        # Logarithmic should have faster initial noise accumulation
        # Check that alpha drops faster early on
        alpha_at_0_1, _ = noise_schedule(torch.tensor(0.1))
        drop_early = 1.0 - alpha_at_0_1

        # Early drop should be significant
        assert drop_early > 0.05

    def test_inverse(self, noise_schedule: LogarithmicNoise) -> None:
        """Test that inverse recovers original timestep."""
        t_original = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        alpha_t, _ = noise_schedule(t_original)
        t_recovered = noise_schedule.inverse(alpha_t)

        assert torch.allclose(t_original, t_recovered, atol=1e-4)


class TestNoiseScheduleComparisons:
    """Compare behavior across different noise schedules."""

    def test_all_schedules_monotonic(self) -> None:
        """Verify all schedules are monotonically decreasing."""
        schedules = [
            LinearNoise(),
            CosineNoise(),
            ExponentialNoise(exp=2),
            LogarithmicNoise(),
        ]

        t_values = torch.linspace(0, 1, 50)
        for schedule in schedules:
            if isinstance(schedule, ExponentialNoise):
                _, alpha_t = schedule(t_values)  # Swapped order
            else:
                alpha_t, _ = schedule(t_values)

            # Verify monotonicity
            for i in range(len(alpha_t) - 1):
                assert alpha_t[i] >= alpha_t[i + 1], f"{schedule.name} not monotonic"

    def test_all_schedules_have_inverse(self) -> None:
        """Verify all schedules implement inverse correctly."""
        schedules = [
            LinearNoise(),
            CosineNoise(),
            ExponentialNoise(exp=2),
            LogarithmicNoise(),
        ]

        t_original = torch.tensor([0.2, 0.5, 0.8])
        for schedule in schedules:
            if isinstance(schedule, ExponentialNoise):
                _, alpha_t = schedule(t_original)
            else:
                alpha_t, _ = schedule(t_original)

            t_recovered = schedule.inverse(alpha_t)
            assert torch.allclose(t_original, t_recovered, atol=1e-3), (
                f"{schedule.name} inverse failed"
            )


@pytest.mark.parametrize("schedule_class", [LinearNoise, CosineNoise, LogarithmicNoise])
def test_batch_processing(schedule_class: type) -> None:
    """Test that schedules handle batch inputs correctly."""
    if schedule_class == LinearNoise:
        noise = schedule_class()
    else:
        noise = schedule_class(eps=1e-3)

    # Test various batch sizes
    for batch_size in [1, 16, 128, 1024]:
        t = torch.rand(batch_size)
        if isinstance(noise, ExponentialNoise):
            _, alpha_t = noise(t)
        else:
            alpha_t, _ = noise(t)

        assert alpha_t.shape == (batch_size,)
        assert torch.all((alpha_t >= 0) & (alpha_t <= 1))
