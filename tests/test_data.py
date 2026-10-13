import numpy as np
import pytest
import torch
from helpers import write_shards

from qtro.data import (
    SweepData,
    fidelity,
    manifest_sha256,
    nested_training_fraction,
    train_validation_split,
    tv,
)


def test_fidelity_of_identical_distributions_is_one():
    p = torch.rand(16)
    p = p / p.sum()
    assert torch.isclose(fidelity(p, p), torch.tensor(1.0), atol=1e-6)


def test_fidelity_of_disjoint_supports_is_zero():
    p = torch.tensor([0.5, 0.5, 0.0, 0.0])
    q = torch.tensor([0.0, 0.0, 0.5, 0.5])
    assert torch.isclose(fidelity(p, q), torch.tensor(0.0), atol=1e-6)


def test_total_variation_bounds():
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])
    assert torch.isclose(tv(p, q), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(tv(p, p), torch.tensor(0.0), atol=1e-6)


def _split_half_histograms():
    torch.manual_seed(0)
    M = 512
    rho_true = torch.rand(M)
    rho_true = rho_true / rho_true.sum()
    draws1 = torch.multinomial(rho_true, 256, replacement=True)
    draws2 = torch.multinomial(rho_true, 256, replacement=True)
    p = torch.zeros(M).index_add_(0, draws1, torch.ones(256)) / 256
    q = torch.zeros(M).index_add_(0, draws2, torch.ones(256)) / 256
    return p, q


def test_fidelity_default_is_exact_on_sparse_high_dimensional_distributions():
    """The default eps=0 is bit-for-bit the epsilon-free Bhattacharyya coefficient.

    Measurement, and the split-half floor above all, must not be biased, and
    any guard inside the square roots biases the value upward.
    """
    p, q = _split_half_histograms()
    exact = (torch.sqrt(p.clamp(min=0)) * torch.sqrt(q.clamp(min=0))).sum(-1) ** 2
    assert torch.isclose(fidelity(p, q), exact, atol=1e-7)
    assert torch.isclose(fidelity(p, q, eps=0.0), exact, atol=1e-7)


def test_fidelity_backward_is_finite_with_a_positive_eps_at_a_single_sided_zero():
    """Training must pass a positive eps: at eps=0 the gradient is infinite
    wherever exactly one of p, q is zero, which split-half histograms contain."""
    p, q = _split_half_histograms()
    mismatch = (p == 0) & (q > 0)
    assert mismatch.any(), "fixture must contain a single-sided-zero cell"

    p0 = p.clone().requires_grad_(True)
    fidelity(p0, q, eps=0.0).backward()
    assert torch.isinf(p0.grad[mismatch]).all()

    p1 = p.clone().requires_grad_(True)
    fidelity(p1, q, eps=1e-12).backward()
    assert torch.isfinite(p1.grad).all()


def test_sweep_data_round_trips_a_shard(tmp_path):
    write_shards(tmp_path)
    data = SweepData(tmp_path, horizons=[1, 2])
    assert len(data) == 4
    b = data.batch(torch.tensor([0, 2]), "cpu")
    assert b["targets"][1].shape[0] == 2
    assert b["targets"][1].dtype == torch.float32
    # Columns of a target are distributions.
    assert torch.allclose(b["targets"][1].sum(1),
                          torch.ones_like(b["targets"][1].sum(1)), atol=1e-2)


def test_sweep_data_rejects_a_horizon_the_shards_lack(tmp_path):
    write_shards(tmp_path, horizons=(1,))
    with pytest.raises(KeyError, match="target_h2"):
        SweepData(tmp_path, horizons=[1, 2])


def test_sweep_data_can_load_an_eval_only_horizon(tmp_path):
    (path,) = write_shards(tmp_path, horizons=(1,))
    d = dict(np.load(path))
    d["ref_h2"] = d["target_h1"]
    d["count_ref_h2"] = d["count_h1"]
    d["noise_fidelity_h2"] = d["noise_fidelity_h1"]
    d["noise_tv_h2"] = np.zeros_like(d["noise_fidelity_h1"])
    np.savez_compressed(path, **d)
    data = SweepData(tmp_path, horizons=[2], want_refs=True, require_targets=False)
    assert 2 not in data.targets
    assert data.refs[2].shape == d["ref_h2"].shape
    batch = data.batch(torch.tensor([0]), "cpu")
    assert batch["targets"] == {}
    assert batch["refs"][2].shape[0] == 1


def test_sweep_data_rejects_an_absent_split(tmp_path):
    write_shards(tmp_path)
    with pytest.raises(ValueError, match="split"):
        SweepData(tmp_path, horizons=[1], splits=("ns3_heldout",))


def test_sweep_data_selects_rows_by_split(tmp_path):
    write_shards(tmp_path, n_conditions=4, splits=("train", "ns3_heldout"))
    assert len(SweepData(tmp_path, horizons=[1], splits=("train",))) == 2
    assert len(SweepData(tmp_path, horizons=[1], splits=("ns3_heldout",))) == 2


def test_training_fraction_is_nested_and_validated():
    indices = torch.tensor([9, 3, 7, 1, 5, 0, 2, 4])
    quarter = nested_training_fraction(indices, .25)
    half = nested_training_fraction(indices, .5)
    whole = nested_training_fraction(indices, 1.0)
    assert torch.equal(quarter, indices[:2])
    assert torch.equal(half, indices[:4])
    assert torch.equal(whole, indices)
    assert torch.equal(quarter, half[:len(quarter)])
    with pytest.raises(ValueError, match="train_fraction"):
        nested_training_fraction(indices, 0.0)


def test_zero_validation_fraction_keeps_every_row():
    order = torch.tensor([3, 1, 0, 2])
    val, train_idx = train_validation_split(order, 0.0)
    assert val.numel() == 0
    assert torch.equal(train_idx, order)
    val, train_idx = train_validation_split(order, 0.25)
    assert torch.equal(val, order[:1])
    assert torch.equal(train_idx, order[1:])


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 2.0])
def test_invalid_validation_fraction_is_rejected(fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        train_validation_split(torch.arange(4), fraction)


def test_manifest_hash_is_none_without_a_manifest(tmp_path):
    assert manifest_sha256(tmp_path) is None
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert len(manifest_sha256(tmp_path)) == 64
