"""Compressed image-shard storage, shared by the perception and policy datasets.

Camera frames dominate both datasets by three orders of magnitude, and raw uint8 in an
npz makes collection I/O-bound and training memory-bound: 128x128x3 is 48 KiB a frame, so
a 100k-frame policy set would be 10 GiB and would not fit in RAM alongside the simulator.
Frames are therefore JPEG-encoded at quality 95 -- about 4 KiB each, visually lossless at
this resolution -- and packed into a single flat byte buffer with an offset table, which
npz stores and mmaps efficiently. Everything else (labels, proprioception, actions) is
stored as ordinary float32 arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

JPEG_QUALITY = 95


def encode_frames(frames: Iterable[np.ndarray], *, quality: int = JPEG_QUALITY) -> tuple[np.ndarray, np.ndarray]:
    """``(N, H, W, 3)`` uint8 RGB -> ``(buffer, offsets)``.

    ``offsets`` has N+1 entries; frame *i* is ``buffer[offsets[i]:offsets[i + 1]]``.
    """
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    blobs: list[np.ndarray] = []
    for frame in frames:
        ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1], params)
        if not ok:  # OpenCV only fails here on a malformed array, never on content
            raise ValueError(f"could not JPEG-encode a frame of shape {frame.shape}")
        blobs.append(buf.reshape(-1))
    sizes = np.array([b.size for b in blobs], dtype=np.int64)
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    buffer = np.concatenate(blobs) if blobs else np.zeros(0, dtype=np.uint8)
    return buffer, offsets


def decode_frame(buffer: np.ndarray, offsets: np.ndarray, i: int) -> np.ndarray:
    """Decode one frame back to ``(H, W, 3)`` uint8 RGB."""
    blob = buffer[offsets[i] : offsets[i + 1]]
    img = cv2.imdecode(blob, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"frame {i} failed to decode ({blob.size} bytes)")
    return img[:, :, ::-1]


def decode_frames(buffer: np.ndarray, offsets: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    """Decode a batch of frames into one ``(len(indices), H, W, 3)`` uint8 array."""
    return np.stack([decode_frame(buffer, offsets, int(i)) for i in indices])


def write_shard(path: str | Path, arrays: dict[str, Any]) -> Path:
    """Write one compressed shard. JPEG buffers are already compressed, so ``savez`` is
    used rather than ``savez_compressed`` -- deflate on JPEG bytes costs seconds and buys
    under 1%. Label arrays are small enough that it does not matter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def read_shard(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as handle:
        return {k: handle[k] for k in handle.files}


def shard_paths(directory: str | Path, prefix: str) -> list[Path]:
    return sorted(Path(directory).glob(f"{prefix}_*.npz"))
