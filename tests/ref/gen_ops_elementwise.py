"""Generate golden data for the elementwise op tests (tests/test_ops_elementwise.c).

The file stores the inputs and the expected outputs. Input values come from a
seeded torch generator; the C test loads them with mlc_from_data. Entry names
must match the C test.

Run from the repository root:
    uv run --project tests/ref python tests/ref/gen_ops_elementwise.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from mlinc_ref.golden import write_golden

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

Unary = Callable[[torch.Tensor], torch.Tensor]
Binary = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Ops whose domain is all real numbers.
UNARY_ANY: dict[str, Unary] = {
    "neg": torch.neg,
    "abs": torch.abs,
    "exp": torch.exp,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    # Exact GELU: x * Phi(x), with the erf formula (PyTorch's default).
    "gelu": lambda x: torch.nn.functional.gelu(x, approximate="none"),
}
# Ops that need positive inputs (log, sqrt).
UNARY_POS: dict[str, Unary] = {"log": torch.log, "sqrt": torch.sqrt}

BINARY: dict[str, Binary] = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
}


def main() -> None:
    g = torch.Generator().manual_seed(1234)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.float32)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=g, dtype=torch.float32)

    def nonzero(*shape: int) -> torch.Tensor:
        """Values with |v| in [0.5, 1.5): safe as divisors."""
        sign = torch.where(rand(*shape) < 0.5, -1.0, 1.0)
        return sign * (rand(*shape) + 0.5)

    d: dict[str, torch.Tensor] = {}

    # ---- unary inputs ----
    d["x"] = randn(3, 5) * 2.0
    d["xp"] = rand(3, 5) * 4.0 + 0.1
    # Extreme values: overflow in exp, saturation in tanh/sigmoid, relu(0).
    d["special"] = torch.tensor([-1000.0, -100.0, -1.0, 0.0, 1.0, 100.0, 1000.0])
    # log/sqrt edge cases: negative -> nan, 0 -> -inf / 0, tiny and huge.
    d["special_pos"] = torch.tensor([-1.0, 0.0, 1e-30, 1.0, 4.0, 1e30])

    for name, fn in UNARY_ANY.items():
        d[f"{name}_x"] = fn(d["x"])
        d[f"{name}_xT"] = fn(d["x"].T)
        d[f"{name}_special"] = fn(d["special"])
    for name, fn in UNARY_POS.items():
        d[f"{name}_xp"] = fn(d["xp"])
        d[f"{name}_xpT"] = fn(d["xp"].T)
        d[f"{name}_special_pos"] = fn(d["special_pos"])

    # ---- binary inputs ----
    d["a_same"] = randn(4, 6)
    d["b_same"] = nonzero(4, 6)
    d["a_bc"] = randn(3, 1, 4)
    d["b_bc"] = nonzero(5, 4)
    d["s"] = nonzero(1).reshape(())  # 0-dim scalar tensor
    d["c_t"] = nonzero(6, 4)  # contiguous partner for a_same.T

    for name, fn in BINARY.items():
        d[f"{name}_same"] = fn(d["a_same"], d["b_same"])
        d[f"{name}_bc"] = fn(d["a_bc"], d["b_bc"])
        d[f"{name}_scalar_right"] = fn(d["a_same"], d["s"])
        d[f"{name}_scalar_left"] = fn(d["s"], d["b_same"])
        d[f"{name}_T"] = fn(d["a_same"].T, d["c_t"])

    # ---- pow ----
    d["pa_same"] = rand(4, 6) * 3.0 + 0.1  # positive bases
    d["pb_same"] = randn(4, 6)
    d["pa_bc"] = rand(3, 1, 4) * 3.0 + 0.1
    d["pb_bc"] = randn(5, 4)
    d["pow_same"] = torch.pow(d["pa_same"], d["pb_same"])
    d["pow_bc"] = torch.pow(d["pa_bc"], d["pb_bc"])
    d["pow_2_x"] = torch.pow(d["x"], 2.0)  # negative bases, integer power
    d["pow_2_5_xp"] = torch.pow(d["xp"], 2.5)
    d["pow_neg_half_xp"] = torch.pow(d["xp"], -0.5)
    d["pow_0_5_special"] = torch.pow(d["special"], 0.5)  # negative -> nan

    # ---- in-place cases ----
    # Add 1 to column 2 of a_same through a [4, 1] slice view.
    col = d["a_same"].clone()
    col[:, 2:3] += 1.0
    d["add_inplace_column"] = col
    # a_same += b_row, with b_row broadcast over the rows.
    d["b_row"] = nonzero(6)
    d["add_inplace_bc"] = d["a_same"] + d["b_row"]
    # Overlap: sq += sq.T must use the old values of sq.T.
    d["sq"] = randn(4, 4)
    d["add_inplace_overlap"] = d["sq"] + d["sq"].T
    # exp_ on every second element of a 1-D tensor.
    d["v10"] = randn(10)
    strided = d["v10"].clone()
    strided[0::2] = torch.exp(strided[0::2])
    d["exp_inplace_step2"] = strided
    # Overlap with the SAME layout but a different offset: v6[1:5] += v6[0:4].
    # Every element must use the old value of its left neighbour.
    d["v6"] = randn(6)
    shifted = d["v6"].clone()
    shifted[1:5] = d["v6"][1:5] + d["v6"][0:4]
    d["add_inplace_shifted_overlap"] = shifted

    write_golden(
        DATA_DIR / "ops_elementwise.mlct", {k: v.contiguous() for k, v in d.items()}
    )
    print(f"wrote {len(d)} entries to {DATA_DIR / 'ops_elementwise.mlct'}")


if __name__ == "__main__":
    main()
