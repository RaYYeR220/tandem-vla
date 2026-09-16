"""Loading the demonstration shards.

Unlike the perception set, the demonstration set is too large to hold decoded: ~60k
transitions at two 128x128 frames each is 5.9 GiB of uint8. The JPEG buffers stay resident
(about 0.8 GiB) and batches are decoded on demand by a background thread, which keeps the
GPU fed without ever materialising the decoded set.

The split is **by episode seed**, so no frame from a validation episode is ever seen during
training -- transitions within one episode are near-duplicates at 16 Hz, and a random frame
split would be close to training on the validation set.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..perception.shards import decode_frame, read_shard, shard_paths

VAL_FRACTION = 0.15


@dataclass
class PolicyData:
    """JPEG-resident demonstration set."""

    shards: list[dict]
    locator: np.ndarray  #: (N, 2) -> shard index, index within shard
    proprio: np.ndarray  #: (N, 12)
    cond: np.ndarray  #: (N, 22)
    chunk: np.ndarray  #: (N, K, 12)
    skill: np.ndarray  #: (N,) str
    arm: np.ndarray  #: (N,) str
    seed: np.ndarray  #: (N,) int32

    def __len__(self) -> int:
        return len(self.seed)

    def subset(self, index: np.ndarray) -> "PolicyData":
        return PolicyData(
            shards=self.shards,
            locator=self.locator[index],
            proprio=self.proprio[index],
            cond=self.cond[index],
            chunk=self.chunk[index],
            skill=self.skill[index],
            arm=self.arm[index],
            seed=self.seed[index],
        )

    def images(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode the overhead and wrist frames for the given sample rows."""
        n = len(rows)
        overhead = np.empty((n, 128, 128, 3), dtype=np.uint8)
        wrist = np.empty_like(overhead)
        for i, row in enumerate(rows):
            shard_index, local = self.locator[row]
            shard = self.shards[shard_index]
            overhead[i] = decode_frame(shard["overhead_jpeg"], shard["overhead_offsets"], local)
            wrist[i] = decode_frame(shard["wrist_jpeg"], shard["wrist_offsets"], local)
        return overhead, wrist


def load(directory: str | Path, *, limit: int | None = None) -> PolicyData:
    paths = shard_paths(directory, "policy")
    if not paths:
        raise FileNotFoundError(f"no policy shards under {directory}")

    shards: list[dict] = []
    locators: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("proprio", "cond", "chunk", "skill",
                                                          "arm", "seed")}
    total = 0
    for shard_index, path in enumerate(paths):
        shard = read_shard(path)
        count = len(shard["seed"])
        shards.append(shard)
        locators.append(
            np.stack([np.full(count, shard_index, dtype=np.int64), np.arange(count)], axis=1)
        )
        for key in parts:
            parts[key].append(shard[key])
        total += count
        if limit is not None and total >= limit:
            break

    data = PolicyData(
        shards=shards,
        locator=np.concatenate(locators),
        proprio=np.concatenate(parts["proprio"]).astype(np.float32),
        cond=np.concatenate(parts["cond"]).astype(np.float32),
        chunk=np.concatenate(parts["chunk"]).astype(np.float32),
        skill=np.concatenate(parts["skill"]),
        arm=np.concatenate(parts["arm"]),
        seed=np.concatenate(parts["seed"]).astype(np.int32),
    )
    if limit is not None and len(data) > limit:
        data = data.subset(np.arange(limit))
    return data


def split_by_seed(data: PolicyData, *, val_fraction: float = VAL_FRACTION
                  ) -> tuple[PolicyData, PolicyData, np.ndarray]:
    seeds = np.unique(data.seed)
    n_val = max(1, int(round(len(seeds) * val_fraction)))
    val_seeds = seeds[-n_val:]
    is_val = np.isin(data.seed, val_seeds)
    return data.subset(~is_val), data.subset(is_val), val_seeds


class BatchStream:
    """Shuffled batches with JPEG decoding overlapped onto a worker thread."""

    def __init__(self, data: PolicyData, batch: int, *, shuffle: bool = True,
                 seed: int = 0, depth: int = 4):
        self.data = data
        self.batch = batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.depth = depth

    def __len__(self) -> int:
        return max(1, len(self.data) // self.batch)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        order = (
            self.rng.permutation(len(self.data)) if self.shuffle else np.arange(len(self.data))
        )
        batches = [order[i * self.batch : (i + 1) * self.batch] for i in range(len(self))]
        pipe: queue.Queue = queue.Queue(maxsize=self.depth)

        def produce() -> None:
            for rows in batches:
                overhead, wrist = self.data.images(rows)
                pipe.put(
                    {
                        "overhead": overhead,
                        "wrist": wrist,
                        "proprio": self.data.proprio[rows],
                        "cond": self.data.cond[rows],
                        "chunk": self.data.chunk[rows],
                    }
                )
            pipe.put(None)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        while True:
            item = pipe.get()
            if item is None:
                break
            yield item
