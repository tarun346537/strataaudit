"""PyTorch temperature scaling on disjoint held-out geological groups."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .constants import DEFAULT_SEED

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class TemperatureFit:
    """Fitted scalar and calibration-set objective values."""

    temperature: float
    calibration_nll_before: float
    calibration_nll_after: float
    observations: int


def group_holdout_split(
    group_ids: Iterable[str],
    seed: int = DEFAULT_SEED,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, str]:
    """Assign complete borehole/region groups, never rows, to distinct splits."""
    groups = sorted(set(group_ids))
    if (
        len(groups) < 3
        or len(fractions) != 3
        or any(f <= 0 for f in fractions)
        or not np.isclose(sum(fractions), 1)
    ):
        raise ValueError(
            "at least three groups and three positive fractions summing to one required"
        )
    order = np.random.default_rng(seed).permutation(len(groups))
    train_count = max(1, min(len(groups) - 2, int(fractions[0] * len(groups))))
    cal_count = max(1, min(len(groups) - train_count - 1, int(fractions[1] * len(groups))))
    return {
        groups[int(j)]: "train"
        if i < train_count
        else "calibration"
        if i < train_count + cal_count
        else "test"
        for i, j in enumerate(order)
    }


def fit_temperature_torch(
    logits: ArrayLike,
    labels: ArrayLike,
    *,
    calibration_groups: Sequence[str],
    training_groups: Sequence[str] = (),
    test_groups: Sequence[str] = (),
    max_iterations: int = 100,
) -> TemperatureFit:
    """Fit temperature on calibration labels after enforcing group independence."""
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(np.asarray(logits, dtype=float), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels, dtype=float), dtype=torch.float64)
    if (
        x.ndim != 1
        or x.shape != y.shape
        or len(x) == 0
        or not torch.isfinite(x).all()
        or not torch.isfinite(y).all()
    ):
        raise ValueError("equal nonempty finite logit/label vectors required")
    if not torch.all((y == 0) | (y == 1)) or len(calibration_groups) != len(x):
        raise ValueError("binary labels and a calibration group for every logit required")
    groups = set(calibration_groups)
    if (
        groups & set(training_groups)
        or groups & set(test_groups)
        or set(training_groups) & set(test_groups)
    ):
        raise ValueError("training, calibration, and test groups must be disjoint")
    if max_iterations < 1:
        raise ValueError("positive optimization iteration count required")
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=max_iterations, line_search_fn="strong_wolfe"
    )
    before = float(F.binary_cross_entropy_with_logits(x, y))

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x / log_temperature.clamp(-6, 6).exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-6, 6).exp())
    after = float(F.binary_cross_entropy_with_logits(x / temperature, y))
    if after > before + 1e-10:  # Guard against optimizer failure, never fit on test labels.
        temperature, after = 1.0, before
    return TemperatureFit(temperature, before, after, len(x))
