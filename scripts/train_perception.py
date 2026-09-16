"""Train the scene-state network on the collected perception shards.

    python scripts/train_perception.py --epochs 60 --batch 128

Reports per-epoch train/validation loss as numbers, then a per-object error table in
millimetres on seeds the network never saw. Writes ``models/perception.pt`` and
``results/perception_train.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tandem.perception import dataset as ds  # noqa: E402
from tandem.perception.geometry import default_cameras, keypoint_targets  # noqa: E402
from tandem.perception.model import PerceptionNet, parameter_count, split_outputs  # noqa: E402
from tandem.perception.schema import (  # noqa: E402
    POS_CENTER,
    POS_SCALE,
    PROPS,
    normalize_pos,
)
from tandem.sim import layout  # noqa: E402


# --------------------------------------------------------------------------- batching


class GpuTensors:
    """Whole dataset resident on the accelerator, batched by index."""

    def __init__(self, data: ds.PerceptionData, device: torch.device, cams: dict):
        self.images = torch.from_numpy(np.ascontiguousarray(data.images)).to(device)
        self.pos = torch.from_numpy(normalize_pos(data.pos)).to(device)
        yaw = torch.from_numpy(data.yaw).to(device)
        self.yaw = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        self.visible = torch.from_numpy(data.visible).to(device)
        self.drawer = torch.from_numpy(data.drawer).to(device)
        # Analytic image-space location of every prop in both views, plus the mask of
        # which of those are worth supervising: a prop that is occluded or outside the
        # frame has no visual evidence at its projected pixel, so pinning attention there
        # would be teaching the network to hallucinate.
        keypoints = keypoint_targets(data.pos, cams)
        inside = (np.abs(keypoints) <= 1.0).all(axis=-1)
        mask = inside & (data.visible[:, None, :] > 0.5)
        self.keypoints = torch.from_numpy(keypoints).to(device)
        self.keypoint_mask = torch.from_numpy(mask.astype(np.float32)).to(device)
        self.n = len(data)

    def batch(self, index: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "images": self.images[index].float().div_(255.0),
            "pos": self.pos[index],
            "yaw": self.yaw[index],
            "visible": self.visible[index],
            "drawer": self.drawer[index],
            "keypoints": self.keypoints[index],
            "keypoint_mask": self.keypoint_mask[index],
        }


def augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Photometric jitter only.

    No crops, flips or shifts: the two cameras are rigidly mounted and the network is
    regressing metric positions in their frame, so any spatial warp would be teaching it a
    camera that does not exist. Brightness, contrast, per-channel gain and sensor noise are
    free to vary and are exactly what changes between two runs of the same cell.
    """
    b = images.shape[0]
    device = images.device
    views = images.view(b, 2, 3, *images.shape[-2:])

    def rand(*shape: int, lo: float, hi: float) -> torch.Tensor:
        return torch.rand(shape, device=device, generator=generator) * (hi - lo) + lo

    views = views * rand(b, 2, 1, 1, 1, lo=0.80, hi=1.20)
    views = views * rand(b, 2, 3, 1, 1, lo=0.93, hi=1.07)
    mean = views.mean(dim=(-1, -2), keepdim=True)
    views = (views - mean) * rand(b, 2, 1, 1, 1, lo=0.85, hi=1.15) + mean
    noise = torch.randn(views.shape, device=device, generator=generator)
    views = views + noise * rand(b, 1, 1, 1, 1, lo=0.0, hi=0.035)
    return views.clamp_(0.0, 1.0).view_as(images)


# ------------------------------------------------------------------------------ losses


def compute_loss(raw: torch.Tensor, target: dict[str, torch.Tensor],
                 weights: dict[str, float],
                 keypoints: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, dict[str, float]]:
    out = split_outputs(raw)
    pos_loss = nn.functional.smooth_l1_loss(out["pos"], target["pos"], beta=0.02)
    yaw_loss = nn.functional.smooth_l1_loss(out["yaw"], target["yaw"], beta=0.1)
    vis_loss = nn.functional.binary_cross_entropy_with_logits(out["vis_logit"], target["visible"])
    drawer_loss = nn.functional.smooth_l1_loss(
        torch.sigmoid(out["drawer_logit"]), target["drawer"], beta=0.05
    )
    total = (
        weights["pos"] * pos_loss
        + weights["yaw"] * yaw_loss
        + weights["vis"] * vis_loss
        + weights["drawer"] * drawer_loss
    )
    parts = {
        "pos": float(pos_loss.detach()),
        "yaw": float(yaw_loss.detach()),
        "vis": float(vis_loss.detach()),
        "drawer": float(drawer_loss.detach()),
    }
    if keypoints is not None:
        mask = target["keypoint_mask"].unsqueeze(-1)
        error = torch.nn.functional.smooth_l1_loss(
            keypoints * mask, target["keypoints"] * mask, beta=0.02, reduction="sum"
        )
        kp_loss = error / mask.sum().clamp(min=1.0) / 2.0
        total = total + weights["keypoint"] * kp_loss
        parts["keypoint"] = float(kp_loss.detach())
    return total, parts


# ----------------------------------------------------------------------------- metrics


@torch.no_grad()
def predict(model: nn.Module, tensors: GpuTensors, batch: int = 256) -> dict[str, np.ndarray]:
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, tensors.n, batch):
        index = torch.arange(start, min(start + batch, tensors.n), device=tensors.images.device)
        images = tensors.images[index].float().div_(255.0)
        chunks.append(model(images).float().cpu())
    raw = torch.cat(chunks)
    out = split_outputs(raw)
    return {
        "pos": out["pos"].numpy(),
        "yaw": out["yaw"].numpy(),
        "vis": torch.sigmoid(out["vis_logit"]).numpy(),
        "drawer": torch.sigmoid(out["drawer_logit"]).numpy(),
    }


def error_table(pred: dict[str, np.ndarray], data: ds.PerceptionData) -> dict:
    """Per-object position error in millimetres, plus drawer and visibility numbers."""
    pos_m = pred["pos"] * POS_SCALE + POS_CENTER
    err_mm = np.linalg.norm(pos_m - data.pos, axis=-1) * 1000.0
    visible = data.visible > 0.5

    per_object = {}
    for i, name in enumerate(PROPS):
        column = err_mm[:, i]
        seen = column[visible[:, i]]
        per_object[name] = {
            "median_mm": round(float(np.median(column)), 2),
            "p90_mm": round(float(np.percentile(column, 90)), 2),
            "mean_mm": round(float(column.mean()), 2),
            "median_mm_visible": round(float(np.median(seen)), 2) if seen.size else None,
            "p90_mm_visible": round(float(np.percentile(seen, 90)), 2) if seen.size else None,
            "visible_frames": int(visible[:, i].sum()),
            "frames": int(column.size),
        }

    yaw_pred = np.arctan2(pred["yaw"][:, :, 1], pred["yaw"][:, :, 0])
    yaw_err = np.abs(np.arctan2(np.sin(yaw_pred - data.yaw), np.cos(yaw_pred - data.yaw)))
    for i, name in enumerate(PROPS):
        seen = np.degrees(yaw_err[visible[:, i], i])
        per_object[name]["yaw_median_deg_visible"] = (
            round(float(np.median(seen)), 1) if seen.size else None
        )

    drawer_abs = np.abs(pred["drawer"] - data.drawer)
    vis_correct = (pred["vis"] > 0.5) == visible
    return {
        "objects": per_object,
        "overall_median_mm": round(float(np.median(err_mm)), 2),
        "overall_p90_mm": round(float(np.percentile(err_mm, 90)), 2),
        "drawer_mae_frac": round(float(drawer_abs.mean()), 4),
        "drawer_mae_mm": round(float(drawer_abs.mean() * layout.DRAWER_OPEN_QPOS * 1000.0), 2),
        "drawer_p90_frac": round(float(np.percentile(drawer_abs, 90)), 4),
        "visibility_accuracy": round(float(vis_correct.mean()), 4),
        "frames": int(len(data)),
    }


# -------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/perception"))
    ap.add_argument("--out", type=Path, default=Path("models/perception.pt"))
    ap.add_argument("--report", type=Path, default=Path("results/perception_train.json"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of frames loaded")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    t0 = time.perf_counter()
    data = ds.load(args.data, limit=args.limit)
    train_data, val_data, val_seeds = ds.split_by_seed(data)
    print(
        f"loaded {len(data)} frames in {time.perf_counter() - t0:.1f}s -> "
        f"{len(train_data)} train / {len(val_data)} val "
        f"(val seeds {int(val_seeds.min())}..{int(val_seeds.max())})"
    )

    cams = default_cameras()
    train = GpuTensors(train_data, device, cams)
    val = GpuTensors(val_data, device, cams)
    del data

    model = PerceptionNet().to(device)
    print(f"PerceptionNet: {parameter_count(model) / 1e6:.2f} M parameters")

    weights = {"pos": 1.0, "yaw": 0.05, "vis": 0.1, "drawer": 0.5, "keypoint": 2.0}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, train.n // args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    history: list[dict] = []
    best = float("inf")

    t_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train.n, device=device, generator=generator)
        running: dict[str, float] = {}
        for step in range(steps_per_epoch):
            index = order[step * args.batch : (step + 1) * args.batch]
            batch = train.batch(index)
            batch["images"] = augment(batch["images"], generator)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                raw, keypoints = model.forward_with_keypoints(batch["images"])
            loss, parts = compute_loss(raw.float(), batch, weights, keypoints.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            running["total"] = running.get("total", 0.0) + float(loss.detach())
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value

        train_loss = {k: round(v / steps_per_epoch, 5) for k, v in running.items()}

        model.eval()
        with torch.no_grad():
            val_running: dict[str, float] = {}
            val_steps = max(1, val.n // 256)
            for step in range(val_steps):
                index = torch.arange(step * 256, min((step + 1) * 256, val.n), device=device)
                batch = val.batch(index)
                raw, keypoints = model.forward_with_keypoints(batch["images"])
                loss, parts = compute_loss(raw.float(), batch, weights, keypoints.float())
                val_running["total"] = val_running.get("total", 0.0) + float(loss)
                for key, value in parts.items():
                    val_running[key] = val_running.get(key, 0.0) + value
        val_loss = {k: round(v / val_steps, 5) for k, v in val_running.items()}

        history.append({"epoch": epoch, "train": train_loss, "val": val_loss,
                        "lr": round(schedule.get_last_lr()[0], 6)})
        if val_loss["total"] < best:
            best = val_loss["total"]
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch}, args.out)
        if epoch % 2 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"epoch {epoch:>3}  train {train_loss['total']:.5f} "
                f"(pos {train_loss['pos']:.5f}, kp {train_loss.get('keypoint', 0):.5f})  "
                f"val {val_loss['total']:.5f} (pos {val_loss['pos']:.5f}, "
                f"kp {val_loss.get('keypoint', 0):.5f}, drawer {val_loss['drawer']:.5f})",
                flush=True,
            )
    train_seconds = time.perf_counter() - t_train

    model.load_state_dict(torch.load(args.out, map_location=device, weights_only=True)["state_dict"])
    metrics = {
        "val": error_table(predict(model, val), val_data),
        "train": error_table(predict(model, train), train_data),
    }

    print("\nHeld-out seeds -- position error per object (millimetres)")
    print(f"{'object':<8} {'median':>8} {'p90':>8} {'median|vis':>11} {'p90|vis':>9} {'vis frames':>11}")
    for name in PROPS:
        row = metrics["val"]["objects"][name]
        print(
            f"{name:<8} {row['median_mm']:>8.1f} {row['p90_mm']:>8.1f} "
            f"{row['median_mm_visible'] or float('nan'):>11.1f} "
            f"{row['p90_mm_visible'] or float('nan'):>9.1f} "
            f"{row['visible_frames']:>7}/{row['frames']}"
        )
    print(
        f"\noverall median {metrics['val']['overall_median_mm']:.1f} mm, "
        f"p90 {metrics['val']['overall_p90_mm']:.1f} mm"
    )
    print(
        f"drawer open-fraction MAE {metrics['val']['drawer_mae_frac']:.4f} "
        f"({metrics['val']['drawer_mae_mm']:.2f} mm of travel), "
        f"visibility accuracy {metrics['val']['visibility_accuracy']:.3f}"
    )

    report = {
        "frames": {"train": len(train_data), "val": len(val_data)},
        "val_seeds": [int(val_seeds.min()), int(val_seeds.max())],
        "parameters": parameter_count(model),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "loss_weights": weights,
        "train_seconds": round(train_seconds, 1),
        "device": str(device),
        "history": history,
        "metrics": metrics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nweights -> {args.out}\nreport  -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
