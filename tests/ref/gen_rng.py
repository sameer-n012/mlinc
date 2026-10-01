"""Generate golden data for the PRNG tests (tests/test_rng.c).

The script first checks the Python reference (mlinc_ref.rng) against two
independent sources, then writes tests/data/rng.mlct. uint64 values are stored
as int64 bit patterns; the C test converts them back to uint64_t.

Run from the repository root:
    uv run --project tests/ref python tests/ref/gen_rng.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mlinc_ref.golden import write_golden
from mlinc_ref.rng import Xoshiro256ss, splitmix64

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Published splitmix64 outputs for seed 1234567 (the reference test vector of
# java.util.SplittableRandom / Vigna's splitmix64.c).
SPLITMIX64_1234567 = [
    6457827717110365317,
    3203168211198807973,
    9817491932198370423,
    4593380528125082431,
    16408922859458223821,
]


def as_i64(values: list[int]) -> npt.NDArray[np.int64]:
    """Store uint64 values as int64 bit patterns."""
    return np.array(values, dtype=np.uint64).view(np.int64)


def check_reference() -> None:
    x = 1234567
    out: list[int] = []
    for _ in range(5):
        x, v = splitmix64(x)
        out.append(v)
    assert out == SPLITMIX64_1234567, "splitmix64 does not match the reference"

    # Cross-check xoshiro256** against the randomgen package (dev dependency).
    from randomgen import Xoshiro256  # pyright: ignore

    bg: Any = Xoshiro256()
    state: Any = bg.state
    state["s"] = np.array([1, 2, 3, 4], dtype=np.uint64)
    state["has_uint32"] = 0
    state["uinteger"] = 0
    bg.state = state
    theirs = [int(v) for v in bg.random_raw(64)]
    rng = Xoshiro256ss([1, 2, 3, 4])
    assert [rng.next() for _ in range(64)] == theirs, "xoshiro256** mismatch"


def main() -> None:
    check_reference()

    data: dict[str, npt.NDArray[Any]] = {
        "splitmix64_1234567": as_i64(SPLITMIX64_1234567),
    }
    rng = Xoshiro256ss([1, 2, 3, 4])
    data["next_state_1_2_3_4"] = as_i64([rng.next() for _ in range(16)])
    for seed in (0, 1, 42):
        rng = Xoshiro256ss.from_seed(seed)
        data[f"state_seed_{seed}"] = as_i64(list(rng.s))
        data[f"next_seed_{seed}"] = as_i64([rng.next() for _ in range(16)])

    rng = Xoshiro256ss.from_seed(7)
    data["randu_seed_7"] = np.array([rng.randu() for _ in range(64)])
    rng = Xoshiro256ss.from_seed(7)
    data["randn_seed_7"] = np.array([rng.randn() for _ in range(64)])

    write_golden(DATA_DIR / "rng.mlct", data)
    print(f"wrote {len(data)} entries to {DATA_DIR / 'rng.mlct'}")


if __name__ == "__main__":
    main()
