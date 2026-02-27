"""Property-based tests for the fidelity gate module.

Tests invariants:
- Identical observed vs simulated data always passes
- Metric computations are bounded and deterministic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


@given(
    n_jobs=st.integers(min_value=20, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_kl_divergence_identical_is_zero(n_jobs: int, seed: int) -> None:
    """KL divergence of identical distributions must be (near) zero."""
    from hpcopt.simulate.metrics import wait_kl_divergence

    rng = np.random.default_rng(seed)
    wait = rng.exponential(scale=300, size=n_jobs).astype(float)

    kl = wait_kl_divergence(wait, wait.copy())
    assert kl < 0.05, f"KL divergence for identical distributions should be near 0, got {kl}"


@given(
    n_jobs=st.integers(min_value=20, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_ks_statistic_identical_is_zero(n_jobs: int, seed: int) -> None:
    """KS statistic of identical distributions must be zero."""
    from hpcopt.simulate.metrics import ks_statistic

    rng = np.random.default_rng(seed)
    values = rng.exponential(scale=300, size=n_jobs).astype(float)

    ks = ks_statistic(values, values.copy())
    assert ks == 0.0, f"KS statistic for identical distributions should be 0, got {ks}"


@given(
    n_jobs=st.integers(min_value=20, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_ks_statistic_bounded(n_jobs: int, seed: int) -> None:
    """KS statistic must always be in [0, 1]."""
    from hpcopt.simulate.metrics import ks_statistic

    rng = np.random.default_rng(seed)
    obs = rng.exponential(scale=300, size=n_jobs).astype(float)
    sim = rng.exponential(scale=600, size=n_jobs).astype(float)

    ks = ks_statistic(obs, sim)
    assert 0.0 <= ks <= 1.0, f"KS statistic out of bounds: {ks}"


@given(
    n_jobs=st.integers(min_value=20, max_value=100),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_queue_series_correlation_identical_is_one(n_jobs: int, seed: int) -> None:
    """Queue series correlation of identical series must be 1.0."""
    from hpcopt.simulate.metrics import queue_series_correlation

    rng = np.random.default_rng(seed)
    ts = np.arange(0, n_jobs * 60, 60)
    queue_len = rng.poisson(lam=10, size=len(ts)).astype(float)

    queue_df = pd.DataFrame({"ts": ts, "queue_len_jobs": queue_len})

    corr = queue_series_correlation(queue_df, queue_df.copy(), int(ts[0]), int(ts[-1]), cadence_sec=60)
    assert abs(corr - 1.0) < 0.01, f"Correlation for identical series should be ~1.0, got {corr}"


def test_empty_arrays_kl() -> None:
    """Empty arrays should return 0 for KL divergence."""
    from hpcopt.simulate.metrics import wait_kl_divergence

    result = wait_kl_divergence(np.array([]), np.array([]))
    assert result == 0.0


def test_empty_arrays_ks() -> None:
    """Empty arrays should return 0 for KS statistic."""
    from hpcopt.simulate.metrics import ks_statistic

    result = ks_statistic(np.array([]), np.array([]))
    assert result == 0.0


def test_relative_divergence_respects_denominator_floor() -> None:
    """Optional denominator floor should reduce low-magnitude ratio noise."""
    from hpcopt.simulate.metrics import relative_divergence

    assert relative_divergence(observed=1.0, simulated=2.0) == 1.0
    assert relative_divergence(observed=1.0, simulated=2.0, denominator_floor=60.0) == 1.0 / 60.0
