"""The whole product in one command: speech in, cutlery on the table.

    python scripts/voice_to_table.py --audio assets/audio/set_the_table.wav --seed 1
    python scripts/voice_to_table.py --mic --seed 3

Streams audio to Speechmatics, takes the final transcript as the instruction, parses it into an
intent locally through OpenVINO, expands that into a plan against the current scene, gates every
step, and runs it on the two arms. Every stage reports which backend actually served it, so a run
with no API key and no model IR still works and still says exactly what it used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # a .env at the repo root is the documented place for the Speechmatics key
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # noqa: BLE001 - the key can equally come from the environment
    pass

from tandem.control.primitives import Executor  # noqa: E402
from tandem.control.runner import EpisodeRunner  # noqa: E402
from tandem.eval.tasks import canonical_plan  # noqa: E402
from tandem.sim.env import TandemEnv  # noqa: E402
from tandem.sim.randomize import RandomizationSpec  # noqa: E402
from tandem.voice.service import VoiceService  # noqa: E402


async def listen(source: str, *, timeout: float = 25.0) -> tuple[str, list[dict], dict]:
    """Stream one utterance and return the joined final transcript plus every event."""
    svc = VoiceService()
    info = svc.info()
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(svc.run(queue, source))
    events: list[dict] = []
    finals: list[str] = []
    last = time.monotonic()
    while True:
        try:
            ev = await asyncio.wait_for(queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            if finals and time.monotonic() - last > 2.5:
                break
            if time.monotonic() - last > timeout:
                break
            continue
        events.append(ev)
        last = time.monotonic()
        if ev["type"] == "partial":
            print(f"    ...{ev.get('text', '')}", flush=True)
        elif ev["type"] == "final":
            finals.append(ev.get("text", "").strip())
            lat = ev.get("latency_ms")
            print(f"  > {ev.get('text', '').strip()}"
                  + (f"   [{float(lat):.0f} ms]" if lat else ""), flush=True)
        elif ev["type"] == "error":
            print(f"  ! {ev.get('detail', '')}", flush=True)
            break
    task.cancel()
    return " ".join(w for w in finals if w).strip(), events, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--audio", type=str, help="WAV file to stream at wall-clock speed")
    src.add_argument("--mic", action="store_true", help="listen on the default microphone")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dr", type=float, default=1.0)
    ap.add_argument("--text", type=str, help="skip listening and use this instruction")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/voice_episode.json")
    args = ap.parse_args()

    record: dict = {"seed": args.seed, "dr_scale": args.dr}

    if args.text:
        instruction, voice_events, voice_info = args.text, [], {"backend": "typed"}
        print(f"\nInstruction (typed): {instruction!r}")
    else:
        source = args.audio or ("mic" if args.mic else "scripted")
        print(f"\nListening [{source}] ...")
        instruction, voice_events, voice_info = asyncio.run(listen(source))
        print(f"\nHeard: {instruction!r}")
        print(f"  via {voice_info.get('backend')} "
              f"(live={voice_info.get('live')})")
    record["voice"] = {"info": voice_info, "instruction": instruction,
                       "events": len(voice_events)}
    if not instruction:
        print("Nothing was transcribed; stopping rather than guessing an instruction.")
        return 1

    env = TandemEnv(RandomizationSpec(scale=args.dr))
    world = env.reset(args.seed)

    print("\nParsing intent ...")
    t0 = time.perf_counter()
    try:
        from tandem.planner import plan as plan_instruction

        plan = plan_instruction(instruction, world)
        meta = plan.get("_meta", {})
    except Exception as exc:  # noqa: BLE001 - the deterministic path is always available
        print(f"  planner unavailable ({exc}); using the deterministic expansion")
        plan, meta = canonical_plan(world), {"backend": "deterministic-expansion"}
    print(f"  backend={meta.get('backend')} device={meta.get('device')} "
          f"precision={meta.get('precision')} in {time.perf_counter() - t0:.2f}s")
    if meta.get("intent"):
        print(f"  intent: {json.dumps(meta['intent'])}")
    record["planner"] = meta

    if plan.get("refusal"):
        print(f"\nREFUSED: {plan['refusal']}")
        record["refusal"] = plan["refusal"]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return 0

    if not plan.get("steps"):
        plan = canonical_plan(world, meta.get("intent"))

    print(f"\nPlan: {len(plan['steps'])} steps")
    for step in plan["steps"]:
        print(f"  {step['id']:>2}. {step['skill']:<12} {json.dumps(step.get('args', {}))}")

    print("\nRunning ...")
    ex = Executor(env)

    def on_event(ev: dict) -> None:
        if ev["type"] == "verdict" and ev["verdict"]["verdict"] != "ALLOW":
            v = ev["verdict"]
            print(f"  GATE {v['verdict']} [{v['code']}] {v['reason']}")
        elif ev["type"] == "step" and ev.get("status") in ("done", "failed"):
            mark = "ok  " if ev["status"] == "done" else "FAIL"
            print(f"  {mark} {ev.get('skill', ''):<12} {ev.get('detail', '')[:64]}")

    rec = EpisodeRunner(env, ex, on_event=on_event).run(
        plan, instruction=instruction, budget_s=args.budget
    )
    score = rec.score
    print("\nResult:")
    for name, done in score["subgoals"].items():
        print(f"  [{'x' if done else ' '}] {name}")
    print(f"  {score['completed']}/{score['total']} subgoals, "
          f"{rec.replans} re-plans, {rec.seconds:.0f}s")

    record["episode"] = rec.as_dict()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"\nWritten to {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
