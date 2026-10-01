"""Python reference for the mlinc PRNG (include/mlc/rng.h).

splitmix64 seeds the state; xoshiro256** generates 64-bit outputs.
mlc_randu maps an output to [0, 1) with 53 bits; mlc_randn uses Box-Muller.
This module and include/mlc/rng.h must use the same algorithms.
"""

from __future__ import annotations

import math

MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> tuple[int, int]:
    """Advance the splitmix64 state `x`. Returns (new_state, output)."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return x, z ^ (z >> 31)


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK64


class Xoshiro256ss:
    """xoshiro256** with the same seeding as mlc_rng_seed."""

    def __init__(self, state: list[int]) -> None:
        if len(state) != 4:
            raise ValueError("state must have 4 words")
        self.s = [v & MASK64 for v in state]

    @classmethod
    def from_seed(cls, seed: int) -> Xoshiro256ss:
        x = seed & MASK64
        words: list[int] = []
        for _ in range(4):
            x, out = splitmix64(x)
            words.append(out)
        return cls(words)

    def next(self) -> int:
        s = self.s
        result = (_rotl((s[1] * 5) & MASK64, 7) * 9) & MASK64
        t = (s[1] << 17) & MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def randu(self) -> float:
        """Uniform in [0, 1): the top 53 bits times 2^-53 (exact in double)."""
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)

    def randn(self) -> float:
        """Standard normal (Box-Muller, one value per call; u1 == 0 retried)."""
        u1 = self.randu()
        while u1 == 0.0:
            u1 = self.randu()
        u2 = self.randu()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
