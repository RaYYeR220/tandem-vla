"""Distil the scripted oracle into the action-chunking policy.

    python scripts/train_policy.py --epochs 40 --batch 64

L1 on the action chunk, which is ACT's standard choice and the right one here: the targets
are joint-position commands whose error distribution has a long tail wherever the oracle
snaps between way-points, and L2 would let those few transitions dominate the gradient.

Reports train/validation loss per epoch as numbers, plus per-skill validation error in
radians and in normalized units. Writes ``models/policy.pt`` and
``results/policy_train.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tandem.policy import dataset as ds  # noqa: E402
from tandem.policy.model import ActionChunkPolicy, parameter_count  # noqa: E402
from tandem.policy.schema import ACTION_DIM, CHUNK, CHUNK_STRIDE  # noqa: E402


def to_device(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for key in ("overhead", "wrist"):
        images = torch.from_numpy(batch[key]).to(device, non_blocking=True)
        out[key] = images.permute(0, 3, 1, 2).float().div_(255.0)
    for key in ("proprio", "cond", "chunk"):
        out[key] = torch.from_numpy(batch[key]).to(device, non_blocking=True)
    return out


def augment(batch: dict[str, torch.Tensor], generator: torch.Generator) -> None:
    """Photometric jitter on both views, in place.

    Same reasoning as the perception net: the cameras are rigid, so no spatial warp, but
    exposure, white balance and sensor noise all move between runs and the policy should
    not care about any of them.
    """
    for key in ("overhead", "wrist"):
        images = batch[key]
        b = images.shape[0]
        device = images.device

        def rand(*shape: int, lo: float, hi: float) -> torch.Tensor:
            return torch.rand(shape, device=device, generator=generator) * (hi - lo) + lo

        images = images * rand(b, 1, 1, 1, lo=0.82, hi=1.18)
        images = images * rand(b, 3, 1, 1, lo=0.94, hi=1.06)
        noise = torch.randn(images.shape, device=device, generator=generator)
        batch[key] = (images + noise * rand(b, 1, 1, 1, lo=0.0, hi=0.03)).clamp_(0.0, 1.0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: ds.PolicyData, device: torch.device,
             batch: int = 128) -> dict:
    """Validation L1, plus per-skill and per-horizon breakdowns."""
    model.eval()
    stream = ds.BatchStream(data, batch, shuffle=False)
    abs_error = np.zeros((CHUNK, ACTION_DIM), dtype=np.float64)
    squared = 0.0
    seen = 0
    per_skill: dict[str, list[float]] = {}
    cursor = 0
    for raw in stream:
        tensors = to_device(raw, device)
        predicted = model(tensors["overhead"], tensors["wrist"], tensors["proprio"],
                          tensors["cond"])
        diff = (predicted - tensors["chunk"]).abs()
        n = diff.shape[0]
        abs_error += diff.sum(0).double().cpu().numpy()
        squared += float((diff**2).sum())
        per_sample = diff.mean(dim=(1, 2)).cpu().numpy()
        for value, skill in zip(per_sample, data.skill[cursor : cursor + n]):
            per_skill.setdefault(str(skill), []).append(float(value))
        cursor += n
        seen += n

    mean_abs = abs_error / max(1, seen)
    return {
        "l1": round(float(mean_abs.mean()), 6),
        "mse": round(squared / max(1, seen * CHUNK * ACTION_DIM), 8),
        "l1_by_horizon": [round(float(v), 6) for v in mean_abs.mean(axis=1)],
        "l1_by_skill": {k: round(float(np.mean(v)), 6) for k, v in sorted(per_skill.items())},
        "samples": seen,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/policy"))
    ap.add_argument("--out", type=Path, default=Path("models/policy.pt"))
    ap.add_argument("--report", type=Path, default=Path("results/policy_train.json"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--chunk-stride", type=int, default=CHUNK_STRIDE,
                    help="control ticks between chunk entries; 1 reproduces the raw recording")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    t0 = time.perf_counter()
    data = ds.restride_chunks(ds.load(args.data, limit=args.limit), args.chunk_stride)
    train_data, val_data, val_seeds = ds.split_by_seed(data)
    skills, counts = np.unique(data.skill, return_counts=True)
    print(
        f"loaded {len(data)} transitions in {time.perf_counter() - t0:.1f}s -> "
        f"{len(train_data)} train / {len(val_data)} val "
        f"(val seeds {int(val_seeds.min())}..{int(val_seeds.max())})"
    )
    print("  by skill: " + ", ".join(f"{s} {c}" for s, c in zip(skills, counts)))

    model = ActionChunkPolicy().to(device)
    print(f"ActionChunkPolicy: {parameter_count(model) / 1e6:.2f} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stream = ds.BatchStream(train_data, args.batch, seed=args.seed)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(stream), pct_start=0.1
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)

    history: list[dict] = []
    best = float("inf")
    t_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, steps = 0.0, 0
        for raw in stream:
            batch = to_device(raw, device)
            augment(batch, generator)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                predicted = model(batch["overhead"], batch["wrist"], batch["proprio"],
                                  batch["cond"])
            loss = torch.nn.functional.l1_loss(predicted.float(), batch["chunk"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            running += float(loss.detach())
            steps += 1

        train_l1 = running / max(1, steps)
        val = evaluate(model, val_data, device)
        history.append({"epoch": epoch, "train_l1": round(train_l1, 6), "val_l1": val["l1"],
                        "val_mse": val["mse"], "lr": round(schedule.get_last_lr()[0], 7)})
        if val["l1"] < best:
            best = val["l1"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, args.out)
        print(
            f"epoch {epoch:>3}  train L1 {train_l1:.5f}  val L1 {val['l1']:.5f}  "
            f"val MSE {val['mse']:.7f}  "
            + "  ".join(f"{k} {v:.5f}" for k, v in val["l1_by_skill"].items()),
            flush=True,
        )
    train_seconds = time.perf_counter() - t_train

    model.load_state_dict(torch.load(args.out, map_location=device, weights_only=True)["state_dict"])
    final_val = evaluate(model, val_data, device)
    final_train = evaluate(model, train_data, device)

    print(f"\nbest checkpoint: val L1 {final_val['l1']:.5f}, train L1 {final_train['l1']:.5f}")
    print("val L1 by chunk step: " + " ".join(f"{v:.4f}" for v in final_val["l1_by_horizon"]))

    report = {
        "transitions": {"train": len(train_data), "val": len(val_data)},
        "val_seeds": [int(val_seeds.min()), int(val_seeds.max())],
        "by_skill": {str(s): int(c) for s, c in zip(skills, counts)},
        "parameters": parameter_count(model),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "chunk": CHUNK,
        "chunk_stride": args.chunk_stride,
        "train_seconds": round(train_seconds, 1),
        "device": str(device),
        "history": history,
        "final": {"val": final_val, "train": final_train},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"weights -> {args.out}\nreport  -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
