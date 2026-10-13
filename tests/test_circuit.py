import torch

from qtro.config import load_config
from qtro.grid import Grid
from qtro.quantum.circuit import ConditionedCircuit
from qtro.quantum.givens import apply_sweep


def _setup(n_layers=2):
    cfg = load_config(None, ["sim.device=cpu"])
    cfg.model.n_layers = n_layers
    grid = Grid(cfg.grid, cfg.net, device="cpu")
    torch.manual_seed(0)
    return grid, ConditionedCircuit(grid, cfg.model, n_cond_features=6)


def test_output_shape():
    grid, circ = _setup()
    psi = circ(torch.randn(4, 6))
    assert psi.shape == (4, grid.M * 4, grid.M)


def test_columns_stay_orthonormal():
    grid, circ = _setup()
    psi = circ(torch.randn(3, 6))
    gram = torch.einsum("bnc,bnd->bcd", psi, psi)
    eye = torch.eye(grid.M).expand(3, grid.M, grid.M)
    assert torch.allclose(gram, eye, atol=1e-4)


def test_conditioning_changes_the_output():
    _, circ = _setup()
    cond = torch.randn(2, 6)
    cond[1] = cond[0] + 1.0
    psi = circ(cond)
    assert not torch.allclose(psi[0], psi[1], atol=1e-3)


def test_identical_conditioning_gives_identical_output():
    _, circ = _setup()
    cond = torch.randn(1, 6).expand(3, 6).contiguous()
    psi = circ(cond)
    assert torch.allclose(psi[0], psi[1], atol=1e-6)
    assert torch.allclose(psi[0], psi[2], atol=1e-6)


def test_conditioning_is_reinjected_at_every_layer():
    """Behavioural, not shape-based: a forward pass that reused one cond_head
    for every layer would pass every shape test. Zero each layer's cond_head
    in turn (restoring it afterwards) and require the output to change every
    time; that is what lets one trained model represent the whole family."""
    _, circ = _setup(n_layers=3)
    assert len(circ.cond_heads) == 3
    for head in circ.cond_heads:
        assert head.shape[0] == circ.n_pairs

    cond = torch.randn(2, 6)
    baseline = circ(cond).detach().clone()

    for layer in range(circ.n_layers):
        head = circ.cond_heads[layer]
        original = head.detach().clone()
        with torch.no_grad():
            head.zero_()
        out = circ(cond)
        with torch.no_grad():
            head.copy_(original)
        assert not torch.allclose(out, baseline, atol=1e-4), layer


def test_is_differentiable():
    _, circ = _setup()
    circ(torch.randn(2, 6)).sum().backward()
    grads = [p.grad for p in circ.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_basis_columns_place_j_tensor_zero_env_in_column_j():
    """Column j of the basis template is |j> tensor |0>_E and nothing else,
    at the real grid size (M=512, d_env=4, n_joint=2048)."""
    grid, circ = _setup(n_layers=1)
    basis = circ._basis_columns(1, torch.device("cpu"), torch.float32)[0]
    assert basis.shape == (grid.M * circ.d_env, grid.M)
    nz_rows, nz_cols = torch.nonzero(basis, as_tuple=True)
    assert torch.equal(nz_cols, torch.arange(grid.M))
    assert torch.equal(nz_rows, torch.arange(grid.M) * circ.d_env)
    assert torch.allclose(basis[nz_rows, nz_cols], torch.ones(grid.M))


def test_per_layer_parameter_counts_match_n_pairs():
    """cond_angles and train_angles each range over all seven sweeps, so both
    parameter sets of every layer hold exactly n_pairs entries (6400 at d_env=4)."""
    grid, circ = _setup(n_layers=3)
    assert circ.n_pairs == 6400
    assert len(circ.trainable) == 3
    assert len(circ.cond_heads) == 3
    for layer_params in circ.trainable:
        assert layer_params.shape == (circ.n_pairs,)
    for head in circ.cond_heads:
        assert head.shape == (circ.n_pairs, 8)


def test_precomputed_untouched_sweep_matches_clone_fallback():
    _, circ = _setup(n_layers=1)
    torch.manual_seed(7)
    psi = torch.randn(2, circ.n_joint, 5, requires_grad=True)
    lo, hi = circ.lo_1, circ.hi_1  # the odd sweep leaves boundary rows untouched
    angles = torch.randn(2, lo.numel(), requires_grad=True)
    legacy = apply_sweep(psi, lo, hi, angles)
    optimised = apply_sweep(psi, lo, hi, angles, circ.untouched_1)
    assert torch.equal(legacy, optimised)

    weight = torch.randn_like(legacy)
    legacy_grads = torch.autograd.grad((legacy * weight).sum(), (psi, angles),
                                       retain_graph=True)
    optimised_grads = torch.autograd.grad(
        (optimised * weight).sum(), (psi, angles))
    assert all(torch.equal(a, b) for a, b in zip(legacy_grads, optimised_grads))


def test_basis_template_is_not_serialised_in_checkpoints():
    _, circ = _setup(n_layers=1)
    assert "basis_columns" not in circ.state_dict()
    assert not any(key.startswith("untouched_") for key in circ.state_dict())
