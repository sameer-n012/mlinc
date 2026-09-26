"""Read and write golden test data files (.mlct).

The format specification is in tests/support/golden.h. This module and the C reader
(tests/support/golden.c) must stay in sync.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

MAGIC = b"MLCT"
VERSION = 1
MAX_NAME = 63
MAX_DIMS = 8

# (numpy kind, item size) -> dtype code in the file.
_DTYPE_CODES: dict[tuple[str, int], int] = {
    ("f", 4): 0,  # f32
    ("f", 8): 1,  # f64
    ("i", 4): 2,  # i32
    ("i", 8): 3,  # i64
}
_CODE_DTYPES: dict[int, np.dtype[Any]] = {
    0: np.dtype("<f4"),
    1: np.dtype("<f8"),
    2: np.dtype("<i4"),
    3: np.dtype("<i8"),
}

Array = npt.NDArray[Any]


def _to_numpy(value: Array | torch.Tensor) -> Array:
    if isinstance(value, torch.Tensor):
        # bf16/f16 tensors have no direct numpy equivalent that the format supports.
        return np.asarray(value.detach().cpu().numpy())
    return value


def write_golden(path: str | Path, tensors: Mapping[str, Array | torch.Tensor]) -> None:
    """Write `tensors` to `path` in the .mlct format.

    Supported dtypes: float32, float64, int32, int64. Cast other dtypes (for example
    bool or float16) before you call this function.
    """
    chunks: list[bytes] = [MAGIC, struct.pack("<II", VERSION, len(tensors))]
    for name, value in tensors.items():
        encoded = name.encode("utf-8")
        if not 1 <= len(encoded) <= MAX_NAME or b"\0" in encoded:
            raise ValueError(f"name {name!r}: must be 1..{MAX_NAME} UTF-8 bytes without NUL")

        arr = _to_numpy(value)
        code = _DTYPE_CODES.get((arr.dtype.kind, arr.dtype.itemsize))
        if code is None:
            raise TypeError(f"{name!r}: dtype {arr.dtype} is not supported; cast it first")
        if arr.ndim > MAX_DIMS:
            raise ValueError(f"{name!r}: ndim {arr.ndim} is larger than {MAX_DIMS}")

        # Row-major, contiguous, little-endian: the layout the C reader expects.
        data = np.ascontiguousarray(arr, dtype=_CODE_DTYPES[code])
        chunks.append(struct.pack("<I", len(encoded)))
        chunks.append(encoded)
        chunks.append(struct.pack(f"<II{arr.ndim}q", code, arr.ndim, *arr.shape))
        chunks.append(data.tobytes())

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"".join(chunks))


def read_golden(path: str | Path) -> dict[str, Array]:
    """Read a .mlct file. Raises ValueError if the file is not valid."""
    buf = Path(path).read_bytes()
    if buf[:4] != MAGIC:
        raise ValueError(f"{path}: bad magic")
    version, count = struct.unpack_from("<II", buf, 4)
    if version != VERSION:
        raise ValueError(f"{path}: version {version} is not supported")

    pos = 12
    result: dict[str, Array] = {}
    try:
        for _ in range(count):
            (name_len,) = struct.unpack_from("<I", buf, pos)
            pos += 4
            name = buf[pos : pos + name_len].decode("utf-8")
            pos += name_len
            code, ndim = struct.unpack_from("<II", buf, pos)
            pos += 8
            shape: tuple[int, ...] = struct.unpack_from(f"<{ndim}q", buf, pos)
            pos += 8 * ndim
            dtype = _CODE_DTYPES[code]
            nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if pos + nbytes > len(buf):
                raise ValueError(f"{path}: entry {name!r} is truncated")
            result[name] = np.frombuffer(buf, dtype=dtype, count=nbytes // dtype.itemsize,
                                         offset=pos).reshape(shape).copy()
            pos += nbytes
    except (struct.error, KeyError) as exc:
        raise ValueError(f"{path}: corrupt file: {exc}") from exc
    if pos != len(buf):
        raise ValueError(f"{path}: unexpected bytes after the last entry")
    return result
