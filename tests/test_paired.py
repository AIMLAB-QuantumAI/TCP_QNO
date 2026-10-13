import numpy as np
import pytest

from qtro.eval.paired import crossed_bootstrap


def test_crossed_bootstrap_is_exact_for_a_constant_advantage():
    stats = crossed_bootstrap(
        np.full((3, 7), .125), n_resamples=200,
        rng=np.random.default_rng(4))
    assert stats["mean_advantage"] == pytest.approx(.125)
    assert stats["mean_ci"] == pytest.approx([.125, .125])
    assert stats["win_rate"] == 1.0
    assert stats["win_rate_ci"] == [1.0, 1.0]
    assert stats["n_seed_config_pairs"] == 21


def test_crossed_bootstrap_ignores_missing_cells_but_counts_them():
    delta = np.full((2, 4), 0.5)
    delta[0, 1] = np.nan
    stats = crossed_bootstrap(delta, n_resamples=50, rng=np.random.default_rng(0))
    assert stats["mean_advantage"] == pytest.approx(0.5)
    assert stats["n_seed_config_pairs"] == 7
    assert (stats["n_seeds"], stats["n_configurations"]) == (2, 4)


def test_crossed_bootstrap_rejects_an_all_nan_matrix():
    with pytest.raises(ValueError, match="finite"):
        crossed_bootstrap(np.full((2, 3), np.nan), n_resamples=10,
                          rng=np.random.default_rng(0))
