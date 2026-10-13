import torch

from qtro.aqm import red_prob


def test_zero_below_min_threshold():
    q = torch.tensor([0.0, 10.0, 19.9])
    assert torch.all(red_prob(q, 0.1, 20.0, 100.0) == 0.0)


def test_reaches_p_max_at_max_threshold():
    p = red_prob(torch.tensor([100.0]), 0.1, 20.0, 100.0)
    assert torch.allclose(p, torch.tensor([0.1]), atol=1e-6)


def test_linear_between_min_and_max():
    p = red_prob(torch.tensor([60.0]), 0.1, 20.0, 100.0)
    assert torch.allclose(p, torch.tensor([0.05]), atol=1e-6)


def test_gentle_region_reaches_one_at_twice_max_threshold():
    p = red_prob(torch.tensor([200.0]), 0.1, 20.0, 100.0)
    assert torch.allclose(p, torch.tensor([1.0]), atol=1e-6)


def test_curve_is_continuous_at_max_threshold():
    below = red_prob(torch.tensor([100.0 - 1e-4]), 0.1, 20.0, 100.0)
    above = red_prob(torch.tensor([100.0 + 1e-4]), 0.1, 20.0, 100.0)
    assert torch.allclose(below, above, atol=1e-5)


def test_monotone_and_bounded():
    q = torch.linspace(0.0, 400.0, 500)
    p = red_prob(q, 0.1, 20.0, 100.0)
    assert torch.all(p[1:] >= p[:-1] - 1e-7)
    assert torch.all(p >= 0.0) and torch.all(p <= 1.0)


def test_broadcasts_over_a_batch_of_controls():
    q = torch.rand(7) * 200.0
    p_max = torch.full((7,), 0.2)
    min_th = torch.full((7,), 10.0)
    p = red_prob(q, p_max, min_th, 100.0)
    assert p.shape == (7,)


def test_gentle_region_slope_at_an_interior_point():
    """Both endpoints of the gentle branch are clamped, so a wrong slope still
    reproduces them; only an interior point distinguishes the denominator."""
    p = red_prob(torch.tensor([150.0]), 0.1, 20.0, 100.0)
    assert torch.allclose(p, torch.tensor([0.55]), atol=1e-6)


def test_zero_below_min_threshold_even_if_thresholds_are_inverted():
    """min_th > max_th is a nonsense control, but it must not mark below min_th."""
    p = red_prob(torch.tensor([0.0, 30.0, 49.9]), 0.1, 50.0, 20.0)
    assert torch.all(p == 0.0)
