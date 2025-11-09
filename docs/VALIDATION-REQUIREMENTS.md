# Validation Requirements for Production Deployment

## Overview

This document outlines the validation steps that **MUST** be completed before deploying the refactored E2D2 codebase to production.

## Critical Gap: Training Validation

⚠️ **IMPORTANT**: While the code has been refactored to production standards, **actual end-to-end training validation has NOT been performed yet**.

### What Has Been Validated ✅

1. **Unit Tests**: Noise schedule modules (98% coverage, 41/41 tests passing)
2. **Import Smoke Tests**: All refactored modules import correctly
3. **Component Tests**: Individual components work in isolation
4. **Code Quality**: All ruff checks pass, 100% type hints in refactored modules
5. **Validation Framework**: OLS regression framework created and tested

### What Has NOT Been Validated ❌

1. **End-to-End Training**: No actual model training has been run
2. **Loss Convergence**: No validation that training actually converges
3. **Integration**: Components haven't been tested together in training loop
4. **Performance**: No validation of training speed/memory usage

## Required Validation Steps

### 1. Minimal Training Smoke Test (CRITICAL)

**Purpose**: Validate that refactored code can train without errors

**Steps**:
```bash
# 1. Create minimal config (tiny model, small dataset)
cp configs/config.yaml configs/smoke_test_config.yaml

# Edit config to use:
# - 2 layer model, 64 hidden dim
# - 100 training samples
# - 100 training steps
# - CPU or single GPU
# - Batch size 4

# 2. Run training
python scripts/composer_scripts/train_discrete_denoiser.py \
    --config-name smoke_test_config \
    training.max_steps=100

# 3. Validate loss convergence
python scripts/smoke_test_training.py --log-file path/to/training_logs.json
```

**Success Criteria**:
- Training completes without errors
- Loss decreases (slope < -0.01)
- Trend is statistically significant (p < 0.05)
- Good linear fit (R² > 0.5)
- Standard error < |slope|

### 2. Full Model Training Test

**Purpose**: Validate production configuration works

**Steps**:
```bash
# Use actual production config
python scripts/composer_scripts/train_discrete_denoiser.py \
    --config-name config \
    training.max_steps=1000
```

**Success Criteria**:
- Training runs for 1000 steps without crashes
- Loss shows clear downward trend
- GPU memory usage is reasonable
- Checkpointing works correctly
- Wandb logging works (if enabled)

### 3. Backward Compatibility Test

**Purpose**: Ensure old configs still work

**Steps**:
```bash
# Test with existing configs from bash_scripts/
bash bash_scripts/run_train_e2d2_gsm8k.sh
```

**Success Criteria**:
- Existing training scripts run without modification
- Results match pre-refactor performance (±5%)

### 4. Refactored Module Integration Test

**Purpose**: Validate new modules integrate correctly

**Steps**:
```bash
# Run smoke tests
python -m pytest tests/smoke/ -v --tb=short

# Run unit tests
python -m pytest tests/unit/ -v --tb=short
```

**Success Criteria**:
- All smoke tests pass
- All unit tests pass
- Coverage > 80% for refactored modules

## OLS Regression Validation

The validation framework uses Ordinary Least Squares regression to statistically validate training convergence:

### Statistical Requirements

| Metric | Requirement | Meaning |
|--------|-------------|---------|
| **Slope** | < -0.01 | Loss must decrease by at least 0.01 per step |
| **P-value** | < 0.05 | Trend must be statistically significant (95% confidence) |
| **R²** | > 0.5 | Linear fit must explain >50% of variance |
| **Std Error** | < \|slope\| | Uncertainty must be less than effect size |

### Interpretation

- **Slope**: Negative slope means loss is decreasing. More negative = faster convergence.
- **P-value**: Low p-value means trend is real, not random noise.
- **R²**: High R² means loss follows a clean linear trend (not too noisy).
- **Std Error**: Low standard error means slope estimate is precise.

### Example Output

```
Regression Results:
  Slope:        -0.014971
  Intercept:    1.9927
  R²:           0.9893
  P-value:      0.000000
  Std Error:    0.000157
  N samples:    101

✅ VALIDATION PASSED: Training converges properly!

Summary:
  • Loss decreases at 0.014971 per step
  • Trend is statistically significant (p=0.000000)
  • Good fit to linear trend (R²=0.9893)
```

## Usage of Validation Tools

### Run Smoke Tests

```bash
# Import and component tests
python -m pytest tests/smoke/ -v

# Specific test
python -m pytest tests/smoke/test_training_convergence.py::test_noise_schedules_work -v
```

### Run OLS Validation

```bash
# With synthetic data (demo)
python scripts/smoke_test_training.py

# With actual training logs (TODO: implement log parsing)
python scripts/smoke_test_training.py --log-file path/to/logs.jsonl
```

## Pre-Deployment Checklist

Before deploying to production, ensure:

- [ ] Minimal training smoke test passes
- [ ] Full model training runs for 1000+ steps
- [ ] Loss convergence validated with OLS regression
- [ ] Backward compatibility confirmed with existing configs
- [ ] All unit tests pass (>80% coverage on refactored modules)
- [ ] All smoke tests pass
- [ ] Ruff checks pass (no linting errors)
- [ ] Mypy type checking passes (on refactored modules)
- [ ] Documentation is complete and accurate
- [ ] Team has reviewed refactored code

## Known Limitations

### Current Status

The refactor has been completed with high code quality standards:
- ✅ Modular architecture (eliminated god classes)
- ✅ Code deduplication (removed 300+ duplicate lines)
- ✅ Structured logging (no debug prints)
- ✅ Type safety (100% coverage in refactored modules)
- ✅ Testing infrastructure (pytest, mypy, ruff)
- ✅ Comprehensive documentation (ADR, README, docstrings)

However:
- ❌ **End-to-end training has not been validated**
- ❌ **Loss convergence has not been statistically confirmed**
- ❌ **Performance benchmarks have not been run**

### Recommended Next Steps

1. **IMMEDIATE**: Run minimal training smoke test (100 steps, CPU)
2. **BEFORE MERGE**: Run full training test (1000 steps, GPU)
3. **BEFORE PRODUCTION**: Run backward compatibility tests
4. **POST-DEPLOYMENT**: Monitor production metrics vs baseline

## Contact

If you encounter issues during validation:

1. Check logs for specific error messages
2. Review ADR-001 for architectural decisions
3. Run smoke tests to isolate the issue
4. File a GitHub issue with validation logs

## Summary

This refactor has established a **solid foundation** for production-grade code:
- Clean architecture
- Comprehensive testing infrastructure
- Statistical validation framework
- Proper documentation

**Critical next step**: Validate that training actually works end-to-end with the refactored code before production deployment.
