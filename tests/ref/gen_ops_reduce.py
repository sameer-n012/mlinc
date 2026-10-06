"""Generate golden data for the reduction tests (tests/test_ops_reduce.c).

Semantics follow PyTorch:
- max/min/median propagate NaN; argmax/argmin return the index of the first
  NaN, else the first occurrence of the extreme value (ties -> lowest index).
- median of an even count is the LOWER of the two middle values.
- argmax/argmin return int64 indices.
Golden values are stored without keepdim; the C test checks keepdim shapes
against the same values. Entry names must match the C test.

Run from the repository root:
    uv run --project tests/ref python tests/ref/gen_ops_reduce.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from mlinc_ref.golden import write_golden

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

Reduce = Callable[[torch.Tensor, int], torch.Tensor]

SINGLE: dict[str, Reduce] = {
    "sum": lambda t, d: torch.sum(t, dim=d),
    "prod": lambda t, d: torch.prod(t, dim=d),
    "mean": lambda t, d: torch.mean(t, dim=d),
    "max": lambda t, d: torch.amax(t, dim=d),
    "min": lambda t, d: torch.amin(t, dim=d),
    "median": lambda t, d: torch.median(t, dim=d).values,
    "argmax": lambda t, d: torch.argmax(t, dim=d),
    "argmin": lambda t, d: torch.argmin(t, dim=d),
}

MultiReduce = Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]

def _prod_dims(t: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """torch.prod takes one dim only: reduce the dims one by one (highest
    first, so the remaining dim numbers stay valid)."""
    for dim in sorted((d % t.ndim for d in dims), reverse=True):
        t = torch.prod(t, dim=dim)
    return t


MULTI: dict[str, MultiReduce] = {
    "sum": lambda t, ds: torch.sum(t, dim=ds),
    "prod": _prod_dims,
    "mean": lambda t, ds: torch.mean(t, dim=ds),
    "max": lambda t, ds: torch.amax(t, dim=ds),
    "min": lambda t, ds: torch.amin(t, dim=ds),
}
MULTI_DIMS: dict[str, tuple[int, ...]] = {
    "d02": (0, 2),
    "dn1_0": (-1, 0),
    "all": (0, 1, 2),
}


def main() -> None:
    g = torch.Generator().manual_seed(4321)
    nan, inf = float("nan"), float("inf")
    d: dict[str, torch.Tensor] = {}

    x = torch.randn(2, 3, 4, generator=g, dtype=torch.float32)
    d["x"] = x
    xt = x.transpose(0, 2)  # [4, 3, 2], non-contiguous in the C test

    for op, fn in SINGLE.items():
        for dim in (0, 1, 2):
            d[f"{op}_d{dim}"] = fn(x, dim)
            d[f"{op}_t_d{dim}"] = fn(xt, dim)

    for op, fn in MULTI.items():
        for name, dims in MULTI_DIMS.items():
            d[f"{op}_{name}"] = fn(x, dims)

    # Ties, NaN, and infinities, reduced over dim 1.
    special = torch.tensor(
        [
            [1.0, nan, 3.0, nan],
            [2.0, 0.0, nan, 1.0],
            [1.0, 3.0, 3.0, 2.0],
            [5.0, 5.0, 5.0, 5.0],
            [-inf, -inf, -inf, -inf],
            [inf, inf, inf, inf],
        ]
    )
    d["special"] = special
    for op in ("max", "min", "median", "argmax", "argmin"):
        d[f"{op}_special"] = SINGLE[op](special, 1)

    write_golden(
        DATA_DIR / "ops_reduce.mlct", {k: v.contiguous() for k, v in d.items()}
    )
    print(f"wrote {len(d)} entries to {DATA_DIR / 'ops_reduce.mlct'}")


if __name__ == "__main__":
    main()
