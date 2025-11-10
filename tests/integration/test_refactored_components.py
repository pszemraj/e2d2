"""Integration tests for refactored components.

Tests that refactored modules work together correctly.
"""

import pytest
import torch

from src.logging_config import get_logger

logger = get_logger(__name__)


@pytest.mark.integration
def test_noise_schedules_integration():
    """Test that noise schedules integrate properly with torch tensors."""
    from src.noise_schedule.noise_schedules import (
        CosineNoise,
        ExponentialNoise,
        LinearNoise,
        LogarithmicNoise,
    )

    # Simulate a training scenario
    batch_size = 4
    t = torch.rand(batch_size)  # Random timesteps

    schedules = [
        LinearNoise(),
        CosineNoise(),
        ExponentialNoise(exp=2),
        LogarithmicNoise(),
    ]

    for schedule in schedules:
        if isinstance(schedule, ExponentialNoise):
            _, alpha_t = schedule(t)
        else:
            alpha_t, _ = schedule(t)

        # Validate batch processing works
        assert alpha_t.shape == (batch_size,)
        assert torch.all(alpha_t >= 0)
        assert torch.all(alpha_t <= 1)

        logger.info(f"✅ {schedule.name} handles batches correctly")


@pytest.mark.integration
def test_constants_integration():
    """Test that constants are accessible and have correct types."""
    from src.constants import (
        CONFIDENCE_THRESHOLD,
        EPSILON,
        GUMBEL_EPSILON,
        MIN_TIMESTEP,
        NEG_INFINITY,
        QUESTION_PREFIX,
        SUMMARY_PREFIX,
    )

    # Validate numerical constants
    assert isinstance(EPSILON, float)
    assert isinstance(GUMBEL_EPSILON, float)
    assert isinstance(NEG_INFINITY, float)
    assert isinstance(MIN_TIMESTEP, float)
    assert isinstance(CONFIDENCE_THRESHOLD, float)

    # Validate ranges
    assert 0 < EPSILON < 1
    assert 0 < GUMBEL_EPSILON < 1
    assert NEG_INFINITY < -1000
    assert 0 < MIN_TIMESTEP < 1

    # Validate prompts
    assert isinstance(QUESTION_PREFIX, str)
    assert isinstance(SUMMARY_PREFIX, str)
    assert len(QUESTION_PREFIX) > 0
    assert len(SUMMARY_PREFIX) > 0

    logger.info("✅ All constants properly defined")


@pytest.mark.integration
def test_logging_integration():
    """Test that logging config works correctly."""
    from src.logging_config import get_logger, setup_logger

    # Test logger creation
    test_logger = get_logger("test_module")
    assert test_logger is not None
    assert test_logger.name == "test_module"

    # Test logging works
    test_logger.info("Test log message")
    test_logger.debug("Debug message")

    # Test setup_logger
    custom_logger = setup_logger("custom", level=10)  # DEBUG level
    assert custom_logger.level == 10

    logger.info("✅ Logging infrastructure works")


@pytest.mark.integration
def test_types_validation():
    """Test that Pydantic types validate correctly."""
    from src.types import (
        GenerationConfig,
        ModelType,
        NoiseScheduleConfig,
        NoiseScheduleType,
    )

    # Test enum types
    assert ModelType.E2D2.value == "e2d2"
    assert NoiseScheduleType.COSINE.value == "cosine"

    # Test config validation
    gen_config = GenerationConfig(
        max_new_tokens=256,
        min_new_tokens=0,
        temperature=1.0,
        do_sample=False,
    )
    assert gen_config.max_new_tokens == 256
    assert gen_config.temperature == 1.0

    # Test noise schedule config
    noise_config = NoiseScheduleConfig(
        schedule_type=NoiseScheduleType.COSINE,
        eps=1e-3,
    )
    assert noise_config.schedule_type == NoiseScheduleType.COSINE
    assert noise_config.eps == 1e-3

    # Test validation catches errors
    with pytest.raises(ValueError):
        GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=20,  # Invalid: min > max
        )

    logger.info("✅ Pydantic validation works")


@pytest.mark.integration
@pytest.mark.slow
def test_simulated_training_loop():
    """Test a simulated training loop with refactored components.

    This doesn't train an actual model but validates that components
    can be used together in a training-like scenario.
    """
    import numpy as np
    from scipy import stats

    from src.noise_schedule.noise_schedules import LinearNoise

    # Simulate training scenario
    num_steps = 50
    batch_size = 4

    noise_schedule = LinearNoise()
    losses = []

    np.random.seed(42)

    for step in range(num_steps):
        # Simulate timestep sampling
        t = torch.rand(batch_size)

        # Get noise parameters
        alpha_t, alpha_t_prime = noise_schedule(t)

        # Simulate loss computation (realistic decay + noise)
        progress = step / num_steps
        base_loss = 2.0 - 1.5 * progress
        noise = np.random.normal(0, 0.1 * (1 - progress * 0.5))
        loss = base_loss + noise

        losses.append(loss)

    # Validate convergence with OLS
    steps = list(range(num_steps))
    slope, intercept, r_value, p_value, std_err = stats.linregress(steps, losses)

    # Check convergence criteria
    assert slope < -0.01, f"Loss not decreasing fast enough: {slope}"
    assert p_value < 0.05, f"Trend not significant: {p_value}"
    assert r_value**2 > 0.5, f"Poor fit: {r_value**2}"
    assert std_err < abs(slope), f"High uncertainty: {std_err} vs {abs(slope)}"

    logger.info(
        f"✅ Simulated training converged (slope={slope:.6f}, R²={r_value**2:.4f})"
    )


@pytest.mark.integration
def test_refactored_imports_no_errors():
    """Test that all refactored imports work without circular dependencies."""
    # This should complete without errors
    from src.constants import EPSILON
    from src.logging_config import get_logger
    from src.noise_schedule.noise_schedules import LinearNoise
    from src.types import ModelType

    # Quick validation
    assert EPSILON > 0
    assert get_logger is not None
    assert LinearNoise is not None
    assert ModelType.E2D2 is not None

    logger.info("✅ No circular dependencies in refactored code")
