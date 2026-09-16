"""Loading the perception shards and splitting them without leaking.

The split is **by seed**, never by frame. Sixteen frames share one seed, and therefore one
draw of lighting, table shade, prop colours, masses and sizes; splitting them at random
would put near-identical appearance conditions on both sides of the wall and turn the
validation number into a memorisation score. The last 15% of seeds, by seed value, are
held out whole.

Frames are decoded once into a contiguous uint8 tensor at load time rather than per batch.
The full 10k-frame set is about 1 GiB, which fits on the training GPU alongside a model
this small, so the input pipeline costs nothing during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import CAMERAS, IMAGE_SIZE, N_PROPS
from .shards import decode_frame, read_shard, shard_paths

VAL_FRACTION = 0.15


@dataclass
class PerceptionData:
    """The whole dataset in memory: images as uint8, labels as float32."""

    images: np.ndarray  #: (N, 6, H, W) uint8 -- overhead RGB then front RGB
    pos: np.ndarray  #: (N, 5, 3) metres
    yaw: np.ndarray  #: (N, 5) radians
    visible: np.ndarray  #: (N, 5) 0/1
    drawer: np.ndarray  #: (N,) open fraction
    seed: np.ndarray  #: (N,) int32
    mode: np.ndarray  #: (N,) str

    def __len__(self) -> int:
        return len(self.seed)

    def subset(self, index: np.ndarray) -> "PerceptionData":
        return PerceptionData(
            images=self.images[index],
            pos=self.pos[index],
            yaw=self.yaw[index],
            visible=self.visible[index],
            drawer=self.drawer[index],
            seed=self.seed[index],
            mode=self.mode[index],
        )


def load(directory: str | Path, *, limit: int | None = None) -> PerceptionData:
    """Read every ``perception_*.npz`` shard under ``directory``."""
    paths = shard_paths(directory, "perception")
    if not paths:
        raise FileNotFoundError(f"no perception shards under {directory}")

    images: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("pos", "yaw", "visible", "drawer",
                                                          "seed", "mode")}
    total = 0
    for path in paths:
        shard = read_shard(path)
        count = len(shard["seed"])
        block = np.empty((count, 6, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for i in range(count):
            for c, cam in enumerate(CAMERAS):
                frame = decode_frame(shard[f"{cam}_jpeg"], shard[f"{cam}_offsets"], i)
                block[i, c * 3 : c * 3 + 3] = np.transpose(frame, (2, 0, 1))
        images.append(block)
        for key in parts:
            parts[key].append(shard[key])
        total += count
        if limit is not None and total >= limit:
            break

    data = PerceptionData(
        images=np.concatenate(images),
        pos=np.concatenate(parts["pos"]).astype(np.float32),
        yaw=np.concatenate(parts["yaw"]).astype(np.float32),
        visible=np.concatenate(parts["visible"]).astype(np.float32),
        drawer=np.concatenate(parts["drawer"]).astype(np.float32),
        seed=np.concatenate(parts["seed"]).astype(np.int32),
        mode=np.concatenate(parts["mode"]),
    )
    if limit is not None and len(data) > limit:
        data = data.subset(np.arange(limit))
    assert data.pos.shape[1] == N_PROPS, "shard prop count disagrees with the schema"
    return data


def held_out_seeds(directory: str | Path, *, val_fraction: float = VAL_FRACTION) -> np.ndarray:
    """The validation seeds, read from the shards' labels without decoding any frames."""
    seeds: list[np.ndarray] = []
    for path in shard_paths(directory, "perception"):
        with np.load(path) as handle:
            seeds.append(handle["seed"])
    unique = np.unique(np.concatenate(seeds))
    n_val = max(1, int(round(len(unique) * val_fraction)))
    return unique[-n_val:]


def load_seeds(directory: str | Path, seeds: np.ndarray, *, limit: int | None = None
               ) -> PerceptionData:
    """Load only the frames belonging to ``seeds``.

    The export and benchmark paths score a few hundred held-out frames; decoding the whole
    30k-frame set to reach them costs three gigabytes of RAM for nothing.
    """
    wanted = set(int(s) for s in seeds)
    images: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("pos", "yaw", "visible", "drawer",
                                                          "seed", "mode")}
    total = 0
    for path in shard_paths(directory, "perception"):
        shard = read_shard(path)
        keep = np.array([int(s) in wanted for s in shard["seed"]])
        rows = np.nonzero(keep)[0]
        if limit is not None:
            rows = rows[: max(0, limit - total)]
        if rows.size == 0:
            continue
        block = np.empty((rows.size, 6, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for out_i, i in enumerate(rows):
            for c, cam in enumerate(CAMERAS):
                frame = decode_frame(shard[f"{cam}_jpeg"], shard[f"{cam}_offsets"], int(i))
                block[out_i, c * 3 : c * 3 + 3] = np.transpose(frame, (2, 0, 1))
        images.append(block)
        for key in parts:
            parts[key].append(shard[key][rows])
        total += rows.size
        if limit is not None and total >= limit:
            break

    if not images:
        raise ValueError(f"no frames under {directory} match the requested seeds")
    return PerceptionData(
        images=np.concatenate(images),
        pos=np.concatenate(parts["pos"]).astype(np.float32),
        yaw=np.concatenate(parts["yaw"]).astype(np.float32),
        visible=np.concatenate(parts["visible"]).astype(np.float32),
        drawer=np.concatenate(parts["drawer"]).astype(np.float32),
        seed=np.concatenate(parts["seed"]).astype(np.int32),
        mode=np.concatenate(parts["mode"]),
    )


def split_by_seed(data: PerceptionData, *, val_fraction: float = VAL_FRACTION
                  ) -> tuple[PerceptionData, PerceptionData, np.ndarray]:
    """Hold out the highest ``val_fraction`` of seeds whole. Returns (train, val, val_seeds)."""
    seeds = np.unique(data.seed)
    n_val = max(1, int(round(len(seeds) * val_fraction)))
    val_seeds = seeds[-n_val:]
    is_val = np.isin(data.seed, val_seeds)
    return data.subset(~is_val), data.subset(is_val), val_seeds
