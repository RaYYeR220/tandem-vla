"""Export both learned nets to OpenVINO and measure what the conversion cost.

    python scripts/export_models.py --iters 200

Produces, per network, IR in FP32, FP16 and NNCF INT8, then two tables in ``results/``:

* ``openvino_latency.{csv,md}`` -- latency and throughput for every
  {precision} x {device} x {LATENCY, THROUGHPUT} combination the host can compile, via
  ``tandem.bench.runner.sweep``.
* ``openvino_accuracy.{json,md}`` -- what each precision did to task quality: per-object
  position error for perception, action MSE against the oracle for the policy, both on the
  same held-out data the training scripts reported on.

INT8 calibration is drawn from **real recorded observations** -- frames out of the
collected datasets, not synthetic noise. Quantizing a vision model on Gaussian noise
calibrates the activation ranges against a distribution the network never sees, and the
resulting scales are wrong in exactly the regions that matter.

Device caveat: this host is not Intel silicon. ``tandem.bench.ovutil.device_report()``
records the actual processor and attaches a caveat string; that caveat is copied into
every artefact this script writes and must stay attached to every number quoted from them.
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

from tandem.bench import ovutil  # noqa: E402
from tandem.bench.runner import sweep, to_csv, to_markdown  # noqa: E402
from tandem.perception import dataset as perception_ds, estimator  # noqa: E402
from tandem.perception.model import PerceptionNet  # noqa: E402
from tandem.perception.schema import (  # noqa: E402
    IMAGE_SIZE,
    POS_CENTER,
    POS_SCALE,
    POS_SLICE,
    PROPS,
    DRAWER_INDEX,
)
from tandem.policy import dataset as policy_ds  # noqa: E402
from tandem.policy.model import ActionChunkPolicy  # noqa: E402
from tandem.policy.schema import ACTION_DIM, CHUNK  # noqa: E402
from tandem.sim import layout  # noqa: E402

import openvino as ov  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- exporting


def export_variants(
    module: torch.nn.Module,
    example: dict[str, torch.Tensor],
    out_dir: Path,
    stem: str,
    calibration: list[dict[str, np.ndarray]],
    *,
    reuse: bool = False,
) -> dict[str, Path]:
    """FP32 + FP16 + NNCF INT8 IR for one network."""
    paths: dict[str, Path] = {p: out_dir / f"{stem}_{p}.xml" for p in ("fp32", "fp16", "int8")}
    if reuse and all(path.exists() for path in paths.values()):
        print(f"  reusing existing {stem} IR")
        for name, path in paths.items():
            print(f"    {name}: {ovutil.detect_precision(path)}, "
                  f"{ovutil.model_size_mb(path):.2f} MiB")
        return paths
    print(f"  exporting {stem} FP32 ...")
    paths["fp32"] = ovutil.export_ir(module, example, out_dir / f"{stem}_fp32.xml", fp16=False)
    print(f"  exporting {stem} FP16 ...")
    paths["fp16"] = ovutil.export_ir(module, example, out_dir / f"{stem}_fp16.xml", fp16=True)
    print(f"  quantizing {stem} INT8 on {len(calibration)} recorded observations ...")
    t0 = time.perf_counter()
    samples = retarget(calibration, ir_input_names(paths["fp32"]))
    paths["int8"] = ovutil.quantize_int8(
        paths["fp32"], samples, out_dir / f"{stem}_int8.xml", subset_size=len(samples)
    )
    print(f"  INT8 done in {time.perf_counter() - t0:.0f}s")
    for name, path in paths.items():
        print(f"    {name}: {ovutil.detect_precision(path)}, {ovutil.model_size_mb(path):.2f} MiB")
    return paths


def _compiled(path: Path, device: str) -> ov.CompiledModel:
    core = ov.Core()
    return core.compile_model(core.read_model(path), device, {"PERFORMANCE_HINT": "LATENCY"})


def ir_input_names(path: Path) -> list[str]:
    """Input names as the IR actually spells them.

    Tracing does not always keep the Python argument name -- a single-input module can come
    back with a numeric tensor name -- so every feed is remapped onto whatever the IR says
    before it is handed to a plugin or to NNCF.
    """
    return [port.get_any_name() for port in ov.Core().read_model(path).inputs]


def retarget(feeds: list[dict[str, np.ndarray]], names: list[str]) -> list[dict[str, np.ndarray]]:
    """Re-key feeds onto ``names``, by name where it matches and by position otherwise."""
    out: list[dict[str, np.ndarray]] = []
    for feed in feeds:
        if all(name in feed for name in names):
            out.append({name: feed[name] for name in names})
        else:
            values = list(feed.values())
            out.append({name: values[i] for i, name in enumerate(names)})
    return out


def _run_batched(path: Path, feeds: list[dict[str, np.ndarray]], device: str) -> np.ndarray:
    compiled = _compiled(path, device)
    names = [port.get_any_name() for port in compiled.inputs]
    request = compiled.create_infer_request()
    outputs = []
    for feed in retarget(feeds, names):
        outputs.append(np.asarray(next(iter(request.infer(feed).values()))))
    return np.concatenate(outputs)


# ------------------------------------------------------------------ perception accuracy


def symbolic_agreement(pos: np.ndarray, drawer: np.ndarray,
                       data: perception_ds.PerceptionData) -> dict:
    """How often the *decisions* derived from the estimate match the simulator's.

    Millimetres are only interesting because the planner and the gate turn them into
    booleans. This runs the estimator's own derivations -- the ones in
    :mod:`tandem.perception.estimator` -- over both the estimated and the true scene and
    reports how often they agree, which is the number that predicts whether the stack
    behaves the same on estimated state as on privileged state.
    """
    agree = {"reachable_by": 0, "on_slot": 0, "in_drawer": 0}
    total = 0
    for frame in range(len(pos)):
        for i, name in enumerate(PROPS):
            estimated, truth = pos[frame, i], data.pos[frame, i]
            agree["reachable_by"] += int(
                layout.reaching_arms(estimated) == layout.reaching_arms(truth)
            )
            agree["on_slot"] += int(
                estimator.on_slot_for(name, estimated) == estimator.on_slot_for(name, truth)
            )
            agree["in_drawer"] += int(
                estimator.in_drawer_for(estimated, drawer[frame])
                == estimator.in_drawer_for(truth, data.drawer[frame])
            )
            total += 1
    threshold = layout.DRAWER_OPEN_THRESHOLD / layout.DRAWER_OPEN_QPOS
    is_open = (drawer >= threshold) == (data.drawer >= threshold)
    return {
        **{k: round(v / max(1, total), 4) for k, v in agree.items()},
        "drawer_is_open": round(float(is_open.mean()), 4),
        "decisions": total,
    }


def perception_metrics(raw: np.ndarray, data: perception_ds.PerceptionData) -> dict:
    pos = raw[:, POS_SLICE].reshape(len(raw), -1, 3) * POS_SCALE + POS_CENTER
    err_mm = np.linalg.norm(pos - data.pos, axis=-1) * 1000.0
    drawer = _sigmoid(raw[:, DRAWER_INDEX])
    per_object = {
        name: {
            "median_mm": round(float(np.median(err_mm[:, i])), 2),
            "p90_mm": round(float(np.percentile(err_mm[:, i], 90)), 2),
        }
        for i, name in enumerate(PROPS)
    }
    return {
        "objects": per_object,
        "symbolic_agreement": symbolic_agreement(pos, drawer, data),
        "overall_median_mm": round(float(np.median(err_mm)), 2),
        "overall_p90_mm": round(float(np.percentile(err_mm, 90)), 2),
        "overall_mean_mm": round(float(err_mm.mean()), 2),
        "drawer_mae_frac": round(float(np.abs(drawer - data.drawer).mean()), 4),
        "drawer_mae_mm": round(
            float(np.abs(drawer - data.drawer).mean() * layout.DRAWER_OPEN_QPOS * 1000.0), 2
        ),
        "frames": int(len(data)),
    }


def verify_estimator(ir_path: Path, seeds: list[int], device: str) -> dict:
    """Run the estimator against live environments and compare it with privileged state.

    The static tables above score the network on recorded frames. This closes the loop:
    the environment is reset, the two cameras are rendered from it, and
    `PerceptionEstimator` produces a whole world-state dict with nothing but those frames
    and the arms' own joint encoders. The dict is then diffed against
    ``TandemEnv.world_state()`` key by key -- which is the actual claim being made, that
    the estimator is a drop-in for privileged state.
    """
    from tandem.perception import PerceptionEstimator, arm_proprioception, render_views
    from tandem.sim.env import TandemEnv
    from tandem.sim.randomize import RandomizationSpec

    env = TandemEnv(RandomizationSpec(scale=1.0))
    est = PerceptionEstimator(ir_path, device=device)
    fields = {"reachable_by": 0, "on_slot": 0, "in_drawer": 0, "held_by": 0}
    latencies: list[float] = []
    errors_mm: list[float] = []
    drawer_errors: list[float] = []
    schema_ok = True
    checks = 0

    for seed in seeds:
        truth = env.reset(seed)
        estimated = est.estimate(render_views(env), arm_proprioception(env),
                                 t=truth["t"], seed=seed)
        latencies.append(est.last_latency_ms)
        schema_ok &= set(truth) <= set(estimated)
        schema_ok &= set(truth["objects"]) == set(estimated["objects"])
        for name in PROPS:
            a, b = estimated["objects"][name], truth["objects"][name]
            for key in fields:
                fields[key] += int(a[key] == b[key])
            errors_mm.append(float(np.linalg.norm(np.array(a["pos"]) - np.array(b["pos"]))) * 1000)
            checks += 1
        drawer_errors.append(abs(estimated["drawer"]["open_frac"] - truth["drawer"]["open_frac"]))

    env.close()
    return {
        "seeds": seeds,
        "schema_superset_of_world_state": bool(schema_ok),
        "field_agreement": {k: round(v / max(1, checks), 4) for k, v in fields.items()},
        "median_pos_err_mm": round(float(np.median(errors_mm)), 2),
        "p90_pos_err_mm": round(float(np.percentile(errors_mm, 90)), 2),
        "drawer_mae_frac": round(float(np.mean(drawer_errors)), 4),
        "mean_infer_ms": round(float(np.mean(latencies)), 3),
    }


# ---------------------------------------------------------------------- policy accuracy


def policy_metrics(raw: np.ndarray, chunks: np.ndarray) -> dict:
    predicted = raw.reshape(len(raw), CHUNK, ACTION_DIM)
    diff = predicted - chunks
    return {
        "action_mse": round(float((diff**2).mean()), 8),
        "action_l1": round(float(np.abs(diff).mean()), 6),
        "action_mse_first_step": round(float((diff[:, 0] ** 2).mean()), 8),
        "samples": int(len(chunks)),
    }


# -------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--perception-data", type=Path, default=Path("data/perception"))
    ap.add_argument("--policy-data", type=Path, default=Path("data/policy"))
    ap.add_argument("--calib", type=int, default=256, help="calibration observations per net")
    ap.add_argument("--accuracy-samples", type=int, default=768,
                    help="held-out samples scored per precision")
    ap.add_argument("--iters", type=int, default=200, help="benchmark iterations per combination")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--verify-seeds", type=int, default=24,
                    help="live environment resets used to verify the drop-in estimator")
    ap.add_argument("--verify-seed0", type=int, default=400)
    ap.add_argument("--reuse-ir", action="store_true",
                    help="benchmark and score the IR already in --models instead of re-exporting")
    ap.add_argument("--skip-policy", action="store_true")
    ap.add_argument("--skip-perception", action="store_true")
    args = ap.parse_args()

    args.models.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    report = ovutil.device_report()
    (args.results / "openvino_devices.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(f"host: {report['processor']}  intel_validated={report['intel_validated']}")
    print(f"devices: {[d['device'] for d in report['devices']]}")
    if "caveat" in report:
        print(f"CAVEAT: {report['caveat']}")

    accuracy: dict[str, dict] = {}
    rows: list[dict] = []

    # ---------------------------------------------------------------- perception
    if not args.skip_perception:
        print("\n== perception ==")
        val_seeds = perception_ds.held_out_seeds(args.perception_data)
        val = perception_ds.load_seeds(args.perception_data, val_seeds,
                                       limit=args.accuracy_samples)
        scored = len(val)

        module = PerceptionNet()
        module.load_state_dict(
            torch.load(args.models / "perception.pt", map_location="cpu",
                       weights_only=True)["state_dict"]
        )
        module.eval()

        images = val.images.astype(np.float32) / 255.0
        feeds = [{"images": images[i : i + 1]} for i in range(scored)]
        calibration = [{"images": images[i : i + 1]} for i in
                       np.linspace(0, scored - 1, min(args.calib, scored)).astype(int)]

        paths = export_variants(module, {"images": torch.zeros(1, 6, IMAGE_SIZE, IMAGE_SIZE)},
                                args.models, "perception", calibration, reuse=args.reuse_ir)

        with torch.no_grad():
            torch_raw = module(torch.from_numpy(images)).numpy()
        accuracy["perception"] = {
            "held_out_seeds": [int(val_seeds.min()), int(val_seeds.max())],
            "frames_scored": scored,
            "torch_fp32": perception_metrics(torch_raw, val),
        }
        for name, path in paths.items():
            raw = _run_batched(path, feeds, "CPU")
            accuracy["perception"][f"ov_{name}"] = {
                **perception_metrics(raw, val),
                "model_mb": ovutil.model_size_mb(path),
                "ir_precision": ovutil.detect_precision(path),
            }
            print(f"  {name}: median {accuracy['perception'][f'ov_{name}']['overall_median_mm']} mm")

        if args.verify_seeds:
            live = list(range(args.verify_seed0, args.verify_seed0 + args.verify_seeds))
            accuracy["perception"]["live_drop_in"] = {
                name: verify_estimator(path, live, "CPU") for name, path in paths.items()
            }
            for name, row in accuracy["perception"]["live_drop_in"].items():
                print(f"  live {name}: median {row['median_pos_err_mm']} mm, "
                      f"reachable_by agreement {row['field_agreement']['reachable_by']:.3f}")

        bench_input = retarget([{"images": images[:1]}], ir_input_names(paths["fp32"]))[0]
        rows += sweep(
            {f"perception_{k}": v for k, v in paths.items()},
            bench_input,
            warmup=args.warmup,
            iters=args.iters,
        )

    # -------------------------------------------------------------------- policy
    if not args.skip_policy:
        print("\n== policy ==")
        data = policy_ds.load(args.policy_data)
        _, val, val_seeds = policy_ds.split_by_seed(data)
        scored = min(args.accuracy_samples, len(val))
        rows_index = np.linspace(0, len(val) - 1, scored).astype(int)
        val = val.subset(rows_index)
        overhead, wrist = val.images(np.arange(len(val)))
        overhead = np.transpose(overhead, (0, 3, 1, 2)).astype(np.float32) / 255.0
        wrist = np.transpose(wrist, (0, 3, 1, 2)).astype(np.float32) / 255.0

        feeds = [
            {
                "overhead": overhead[i : i + 1],
                "wrist": wrist[i : i + 1],
                "proprio": val.proprio[i : i + 1],
                "cond": val.cond[i : i + 1],
            }
            for i in range(len(val))
        ]
        calibration = [feeds[i] for i in
                       np.linspace(0, len(feeds) - 1, min(args.calib, len(feeds))).astype(int)]

        module = ActionChunkPolicy()
        module.load_state_dict(
            torch.load(args.models / "policy.pt", map_location="cpu",
                       weights_only=True)["state_dict"]
        )
        module.eval()

        example = {
            "overhead": torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            "wrist": torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            "proprio": torch.zeros(1, val.proprio.shape[1]),
            "cond": torch.zeros(1, val.cond.shape[1]),
        }
        paths = export_variants(module, example, args.models, "policy", calibration,
                                reuse=args.reuse_ir)

        with torch.no_grad():
            torch_raw = module(
                torch.from_numpy(overhead), torch.from_numpy(wrist),
                torch.from_numpy(val.proprio), torch.from_numpy(val.cond)
            ).numpy()
        accuracy["policy"] = {
            "held_out_seeds": [int(val_seeds.min()), int(val_seeds.max())],
            "samples_scored": scored,
            "torch_fp32": policy_metrics(torch_raw.reshape(scored, -1), val.chunk),
        }
        for name, path in paths.items():
            raw = _run_batched(path, feeds, "CPU")
            accuracy["policy"][f"ov_{name}"] = {
                **policy_metrics(raw.reshape(scored, -1), val.chunk),
                "model_mb": ovutil.model_size_mb(path),
                "ir_precision": ovutil.detect_precision(path),
            }
            print(f"  {name}: action MSE {accuracy['policy'][f'ov_{name}']['action_mse']}")

        bench_input = retarget([feeds[0]], ir_input_names(paths["fp32"]))[0]
        rows += sweep(
            {f"policy_{k}": v for k, v in paths.items()},
            bench_input,
            warmup=args.warmup,
            iters=args.iters,
        )

    # ------------------------------------------------------------------- reports
    baseline = {
        row["model"].split("_")[0]: row["latency_ms"]["mean"]
        for row in rows
        if not row.get("skipped") and row["device"] == "CPU" and row["hint"] == "LATENCY"
        and row["model"].endswith("_fp32")
    }
    for row in rows:
        if row.get("skipped"):
            continue
        reference = baseline.get(row["model"].split("_")[0])
        if reference:
            row["speedup_vs_fp32_cpu"] = round(reference / row["latency_ms"]["mean"], 3)

    to_csv(rows, args.results / "openvino_latency.csv")
    markdown = [
        "# OpenVINO latency and throughput",
        "",
        f"Host: `{report['processor']}` -- Intel-validated: **{report['intel_validated']}**",
        "",
        f"> {report.get('caveat', 'Host is Intel silicon.')}",
        "",
        f"Measured with `tandem.bench.runner.sweep`, {args.iters} iterations after "
        f"{args.warmup} warm-up runs, batch 1.",
        "",
        to_markdown(rows),
    ]
    (args.results / "openvino_latency.md").write_text("\n".join(markdown), encoding="utf-8")

    accuracy["_host"] = report
    (args.results / "openvino_accuracy.json").write_text(
        json.dumps(accuracy, indent=2, default=str), encoding="utf-8"
    )
    (args.results / "openvino_accuracy.md").write_text(
        _accuracy_markdown(accuracy, report), encoding="utf-8"
    )
    print(f"\nwrote {args.results}/openvino_latency.csv|.md and openvino_accuracy.json|.md")
    return 0


def _accuracy_markdown(accuracy: dict, report: dict) -> str:
    lines = [
        "# Precision vs task quality",
        "",
        f"Host: `{report['processor']}` -- Intel-validated: **{report['intel_validated']}**",
        "",
        f"> {report.get('caveat', 'Host is Intel silicon.')}",
        "",
        "All numbers on held-out seeds neither network trained on. INT8 is NNCF "
        "post-training quantization calibrated on recorded observations from the same "
        "datasets.",
        "",
    ]
    variants = [("torch_fp32", "PyTorch FP32"), ("ov_fp32", "OpenVINO FP32"),
                ("ov_fp16", "OpenVINO FP16"), ("ov_int8", "OpenVINO INT8")]

    if "perception" in accuracy:
        block = accuracy["perception"]
        lines += [
            f"## Perception ({block['frames_scored']} held-out frames)",
            "",
            "| variant | IR precision | size (MiB) | median err (mm) | p90 err (mm) | "
            "drawer MAE (mm) | " + " | ".join(f"{p} median" for p in PROPS) + " |",
            "| --- | --- | --- | --- | --- | --- | " + " | ".join("---" for _ in PROPS) + " |",
        ]
        for key, label in variants:
            row = block.get(key)
            if not row:
                continue
            per_object = " | ".join(f"{row['objects'][p]['median_mm']:.1f}" for p in PROPS)
            lines.append(
                f"| {label} | {row.get('ir_precision', 'FP32')} | "
                f"{row.get('model_mb', float('nan')):.2f} | {row['overall_median_mm']:.1f} | "
                f"{row['overall_p90_mm']:.1f} | {row['drawer_mae_mm']:.2f} | {per_object} |"
            )
        live = block.get("live_drop_in")
        if live:
            lines += [
                "",
                f"Live drop-in check: the environment is reset on {len(next(iter(live.values()))['seeds'])} "
                "held-out seeds, the two cameras are rendered from it, and the estimator builds a "
                "whole world-state dict from those frames plus the arms' own joint encoders. "
                "Agreement is against `TandemEnv.world_state()` on the same instant.",
                "",
                "| variant | schema match | reachable_by | on_slot | in_drawer | held_by | "
                "median err (mm) | drawer MAE |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for key, label in (("fp32", "OpenVINO FP32"), ("fp16", "OpenVINO FP16"),
                               ("int8", "OpenVINO INT8")):
                row = live.get(key)
                if not row:
                    continue
                agree = row["field_agreement"]
                lines.append(
                    f"| {label} | {row['schema_superset_of_world_state']} | "
                    f"{agree['reachable_by']:.3f} | {agree['on_slot']:.3f} | "
                    f"{agree['in_drawer']:.3f} | {agree['held_by']:.3f} | "
                    f"{row['median_pos_err_mm']:.1f} | {row['drawer_mae_frac']:.4f} |"
                )
        lines += [
            "",
            "Symbolic agreement -- how often the booleans the planner and gate actually read "
            "come out the same from the estimate as from privileged state, over every "
            "(frame, object) pair:",
            "",
            "| variant | reachable_by | on_slot | in_drawer | drawer.is_open |",
            "| --- | --- | --- | --- | --- |",
        ]
        for key, label in variants:
            row = block.get(key)
            if not row or "symbolic_agreement" not in row:
                continue
            agree = row["symbolic_agreement"]
            lines.append(
                f"| {label} | {agree['reachable_by']:.3f} | {agree['on_slot']:.3f} | "
                f"{agree['in_drawer']:.3f} | {agree['drawer_is_open']:.3f} |"
            )
        lines.append("")

    if "policy" in accuracy:
        block = accuracy["policy"]
        lines += [
            f"## Policy ({block['samples_scored']} held-out transitions)",
            "",
            "| variant | IR precision | size (MiB) | action MSE | action L1 | "
            "first-step MSE |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for key, label in variants:
            row = block.get(key)
            if not row:
                continue
            lines.append(
                f"| {label} | {row.get('ir_precision', 'FP32')} | "
                f"{row.get('model_mb', float('nan')):.2f} | {row['action_mse']:.3e} | "
                f"{row['action_l1']:.5f} | {row['action_mse_first_step']:.3e} |"
            )
        lines.append("")
    lines += [
        "Closed-loop task success at FP32 vs INT8 is measured separately by "
        "`scripts/eval_policy.py --precision fp32,int8`, which runs the same held-out "
        "seeds through the simulator under each IR.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
