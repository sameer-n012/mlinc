"""Generate the golden files for the self-test of the C golden reader (tests/test_golden.c).

Run from the repository root:
    uv run --project tests/ref python tests/ref/gen_golden_selftest.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mlinc_ref.golden import read_golden, write_golden

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    valid = DATA_DIR / "golden_selftest.mlct"
    # These values must match the checks in tests/test_golden.c.
    tensors = {
        "scalar_f32": np.array(3.5, dtype=np.float32),
        "arange_2x3_f32": torch.arange(6, dtype=torch.float32).reshape(2, 3) * 0.5,
        "f64": np.array([1e-300, np.pi], dtype=np.float64),
        "i32": np.array([-1, 2**31 - 1], dtype=np.int32),
        "i64": np.array([-(2**62), 2**40], dtype=np.int64),
        "empty_0x4_f32": np.zeros((0, 4), dtype=np.float32),
        # A transposed view: the writer must store it in row-major order.
        "transposed_3x2_f32": np.arange(6, dtype=np.float32).reshape(2, 3).T,
    }
    write_golden(valid, tensors)

    # Round trip in Python, to check the writer before the C reader uses the file.
    loaded = read_golden(valid)
    assert loaded.keys() == tensors.keys()
    for name, expected in tensors.items():
        exp = expected.numpy() if isinstance(expected, torch.Tensor) else expected
        np.testing.assert_array_equal(loaded[name], exp, err_msg=name)

    # Invalid files for the error-path tests.
    raw = valid.read_bytes()
    (DATA_DIR / "golden_bad_magic.mlct").write_bytes(b"XXXX" + raw[4:])
    (DATA_DIR / "golden_truncated.mlct").write_bytes(raw[:-3])
    (DATA_DIR / "golden_trailing.mlct").write_bytes(raw + b"\0")

    print(f"wrote golden self-test files to {DATA_DIR}")


if __name__ == "__main__":
    main()
