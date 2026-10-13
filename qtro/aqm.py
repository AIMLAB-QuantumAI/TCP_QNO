"""Gentle RED's marking probability, shared by the fluid baselines."""
from __future__ import annotations

import torch
from torch import Tensor


def red_prob(avg_q: Tensor, p_max, min_th, max_th) -> Tensor:
    """Gentle RED marking probability.

    Zero below min_th, linear to p_max at max_th, then linear from p_max to 1
    between max_th and 2 * max_th. The gentle region is what keeps the queue
    reachable above q_crit: a hard drop-all at max_th would leave the tail
    region empty and make the overflow probability unmeasurable.
    """
    avg_q = torch.as_tensor(avg_q)
    p_max = torch.as_tensor(p_max, dtype=avg_q.dtype, device=avg_q.device)
    min_th = torch.as_tensor(min_th, dtype=avg_q.dtype, device=avg_q.device)
    max_th = torch.as_tensor(max_th, dtype=avg_q.dtype, device=avg_q.device)

    low = (avg_q - min_th) / (max_th - min_th) * p_max
    high = p_max + (avg_q - max_th) / max_th * (1.0 - p_max)
    marked = torch.where(avg_q < max_th, low, high)
    # Zero below min_th is asserted here rather than left to the clamp: if a
    # control sweep ever produced min_th > max_th, the high branch would be
    # selected below min_th and return a positive probability.
    return torch.where(avg_q < min_th, torch.zeros_like(marked),
                       marked).clamp(0.0, 1.0)
