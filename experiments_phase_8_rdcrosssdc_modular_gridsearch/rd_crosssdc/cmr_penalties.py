"""Selectable per-sample CMR objectives."""

import math

import torch
from torch.nn import functional as F


CMR_PENALTIES = ("hinge", "direct", "exp", "softplus", "log1p")


def cmr_penalty_per_sample(
    current_margin: torch.Tensor,
    reference_margin: torch.Tensor,
    *,
    penalty: str = "hinge",
    tolerance: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Return a loss vector of shape (B,), without weighting or reduction."""
    if penalty not in CMR_PENALTIES:
        raise ValueError("Unknown CMR penalty: {!r}".format(penalty))

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("CMR scale must be finite and positive")

    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("CMR tolerance must be finite and non-negative")

    if current_margin.ndim != 1 or current_margin.shape != reference_margin.shape:
        raise ValueError(
            "Current and reference margins must both have shape (B,)"
        )

    if current_margin.device != reference_margin.device:
        raise ValueError(
            "Current and reference margins must use the same device"
        )

    if (
        not current_margin.is_floating_point()
        or not reference_margin.is_floating_point()
    ):
        raise TypeError("Margins must be floating-point tensors")

    cur = current_margin

    # Preserve float64; promote low-precision inputs for penalty computation.
    if cur.dtype in (torch.float16, torch.bfloat16):
        cur = cur.float()

    # Teacher/reference is fixed. Student margin retains its gradient.
    ref = reference_margin.detach().to(dtype=cur.dtype)

    if penalty == "direct":
        values = -cur

    else:
        # Signed gap: do not apply ReLU before exp or softplus.
        gap = ref - cur - tolerance

        if penalty == "hinge":
            values = F.relu(gap)

        elif penalty == "exp":
            values = scale * torch.exp(gap / scale)

        elif penalty == "softplus":
            values = scale * F.softplus(gap / scale)

        else:  # log1p
            values = scale * torch.log1p(F.relu(gap) / scale)

    if not torch.isfinite(values).all().item():
        raise FloatingPointError(
            "CMR {!r} produced NaN/Inf. Check margins; for exp, increase "
            "--rd_cmr_scale. The exponent is intentionally not clipped."
            .format(penalty)
        )

    return values