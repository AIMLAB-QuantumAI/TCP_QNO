import torch

from qtro.device import resolve_device


def test_auto_picks_cuda_only_when_available():
    assert resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")


def test_explicit_names_pass_through():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"
