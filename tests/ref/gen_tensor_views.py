"""Generate golden data for the tensor view tests (tests/test_tensor_view.c).

Every case starts from base = arange(24).reshape(2, 3, 4) and applies one or
more view operations. The file stores the result as a contiguous array, so the
C test can compare values and shape. The names must match the C test.

Run from the repository root:
    uv run --project tests/ref python tests/ref/gen_tensor_views.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from mlinc_ref.golden import write_golden

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    col = torch.arange(3, dtype=torch.float32).reshape(3, 1)

    cases = {
        "permute_2_0_1": base.permute(2, 0, 1),
        "transpose_0_2": base.transpose(0, 2),
        "transpose_neg": base.transpose(-1, -2),
        "slice_d1_1_3": base[:, 1:3, :],
        "slice_d2_step2": base[:, :, ::2],
        "slice_neg_start": base[:, :, -3:],
        "slice_then_transpose": base[:, 1:3, :].transpose(-1, -2),
        "slice_then_view": base[1:2].reshape(12),
        "expand_3x1_to_2x3x4": col.expand(2, 3, 4),
        "unsqueeze_then_permute": base.unsqueeze(1).permute(3, 1, 0, 2),
        "squeeze_after_slice": base[:, 1:2, :].squeeze(1),
        "reshape_of_transpose": base.transpose(0, 2).reshape(6, 4),
    }
    write_golden(
        DATA_DIR / "tensor_views.mlct",
        {name: t.contiguous() for name, t in cases.items()},
    )
    print(f"wrote {len(cases)} cases to {DATA_DIR / 'tensor_views.mlct'}")


if __name__ == "__main__":
    main()
