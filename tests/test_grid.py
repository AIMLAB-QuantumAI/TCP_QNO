import torch

from qtro.config import load_config
from qtro.grid import Grid


def _grid():
    cfg = load_config(None, None)
    return Grid(cfg.grid, cfg.net, device="cpu")


def test_size_is_512():
    g = _grid()
    assert g.M == 512
    assert g.w_rep.shape == (512,)
    assert g.q_rep.shape == (512,)
    assert g.phase_rep.shape == (512,)


def test_edges_are_strictly_increasing_and_span_the_range():
    g = _grid()
    assert torch.all(g.w_edges[1:] > g.w_edges[:-1])
    assert torch.all(g.q_edges[1:] > g.q_edges[:-1])
    assert torch.isclose(g.w_edges[0], torch.tensor(1.0))
    assert torch.isclose(g.w_edges[-1], torch.tensor(64.0))
    assert torch.isclose(g.q_edges[0], torch.tensor(0.0))
    assert torch.isclose(g.q_edges[-1], torch.tensor(g.cfg.buffer_pkts))


def test_queue_grid_is_finer_near_q_crit():
    g = _grid()
    widths = g.q_edges[1:] - g.q_edges[:-1]
    centres = 0.5 * (g.q_edges[1:] + g.q_edges[:-1])
    nearest = int(torch.argmin((centres - g.q_crit).abs()))
    assert widths[nearest] < widths.mean()


def test_cell_index_is_a_bijection():
    g = _grid()
    a = torch.arange(g.n_w).repeat_interleave(g.n_q * g.n_phase)
    b = torch.arange(g.n_q).repeat_interleave(g.n_phase).repeat(g.n_w)
    c = torch.arange(g.n_phase).repeat(g.n_w * g.n_q)
    idx = g.cell_index(a, b, c)
    assert torch.equal(torch.sort(idx).values, torch.arange(g.M))


def test_axis_bins_inverts_cell_index():
    g = _grid()
    a, b, c = g.axis_bins()
    assert a.shape == (g.M,) and b.shape == (g.M,) and c.shape == (g.M,)
    assert torch.equal(g.cell_index(a, b, c), torch.arange(g.M))
    assert torch.equal(c, g.phase_rep)


def test_digitize_round_trips_representative_points():
    g = _grid()
    idx = g.digitize(g.w_rep, g.q_rep, g.phase_rep)
    assert torch.equal(idx, torch.arange(g.M))


def test_digitize_clamps_out_of_range_inputs():
    g = _grid()
    w = torch.tensor([0.01, 1e6])
    q = torch.tensor([-5.0, 1e6])
    ph = torch.tensor([0, 1])
    idx = g.digitize(w, q, ph)
    assert torch.all(idx >= 0) and torch.all(idx < g.M)


def test_rtt_increases_with_queue():
    g = _grid()
    order = torch.argsort(g.q_rep)
    assert torch.all(g.rtt_rep[order][1:] >= g.rtt_rep[order][:-1] - 1e-9)


def test_tail_mask_is_nonempty_and_matches_q_crit():
    g = _grid()
    assert g.tail_mask.any()
    assert not g.tail_mask.all()
    assert torch.all(g.q_rep[g.tail_mask] >= g.q_crit)
