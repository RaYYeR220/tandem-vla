# Tandem

**Two robot arms set a dinner table from a spoken instruction — and refuse the ones they can't ground.**

[Overview](https://rayyer220.github.io/tandem-vla/) · [Review in five minutes](JUDGES.md) · [Proof](PROOF.md) · [Honest limits](MOCKS.md)

Tandem is an end-to-end bimanual manipulation stack built on two simulated
[SO-101](https://github.com/TheRobotStudio/SO-ARM100) arms in MuJoCo. You say what you want.
A language model parses the intent. A deterministic gate checks every planned step against the
measured state of the scene and either allows it, repairs it, or refuses it with a reason. Then
the arms do it — handing objects to each other, because neither arm can reach both the props
and the place setting.

Every neural model in the loop — speech, language, perception, policy — runs locally through
**OpenVINO**. Nothing in the control path touches the network.

```
   speech  ──▶  intent  ──▶  plan  ──▶  ┌──────────┐  ──▶  skills  ──▶  two SO-101 arms
 Speechmatics   OpenVINO     state-      │   GATE   │       OpenVINO       MuJoCo
   realtime       LLM        aware       │ allow /  │        policy
                            expansion    │ rewrite/ │
                                         │ refuse   │
                                         └──────────┘
```

---

## The part most demos skip

A bimanual VLA demo is easy to film once. The hard part is what happens when the instruction
doesn't match the world — when someone asks for a knife that isn't on the table, or asks the
right arm to pick up something only the left arm can reach, or asks to pour before anything is
holding the cup.

In Tandem the language model is never trusted with that question. It proposes; a deterministic
gate decides:

| verdict | what it means | example |
| --- | --- | --- |
| `ALLOW` | preconditions hold and the arm can physically reach it | `pick(left, plate)` |
| `REWRITE` | the goal is fine, the assignment isn't — repaired in place | `place(left, plate, slot_plate)` → hand-off + place by the right arm |
| `REFUSE` | cannot be done, and the operator is told why | *"there is no knife in this scene; it holds bottle, fork, mug, plate, spoon"* |

The gate is graded like any other component, with a **negative control** so the score can't be
gamed by a gate that just refuses everything:

```
python scripts/eval_gate.py        # ~1 second, no models, no GPU, no credentials
```

See [`PROOF.md`](PROOF.md) for the current scorecard.

---

## The workcell

The geometry is the point. Props start in the **left** arm's reach; the place setting is in the
**right** arm's. Neither arm can complete a single item alone, so a hand-off is structural
rather than decorative.

- Two SO-101 arms, bases 0.40 m apart, mounted facing the table.
- A cabinet with a sliding drawer holding the cutlery. Shut, a lip covers it — the drawer has
  to be opened before the fork and spoon exist as far as the arms are concerned.
- Props: a deep plate, a tumbler, an open carton, a fork and a spoon.
- Ten water particles inside the carton, so a pour is measured by **how much liquid actually
  lands in the mug**, not by whether the wrist rotated.
- Cameras: overhead, front, cinematic, and a wrist camera on each arm.

Everything is sized against the **measured** SO-101 gripper: the fingertips travel 133 mm apart,
but the jaw faces close in well ahead of them, so the real usable opening is about 50 mm. Every
prop fits inside that, and the numbers in `tandem/sim/layout.py` say where they came from.

---

## Quick start

```bash
git clone <this repo> && cd tandem
python -m venv .venv && .venv/Scripts/activate       # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

pytest tandem/ -q                                     # 81 tests, ~13 s
python -m tandem.sim.scene                            # compile the cell, print its dimensions
python scripts/eval_gate.py                           # grade the safety gate (1 s)
python scripts/evaluate.py --seeds 20 --out results/eval   # the task scorecard
python scripts/benchmark_intel.py                     # OpenVINO device + latency report
python scripts/serve.py                               # the live dashboard on :8000
```

Nothing above needs an API key, a GPU, or a model download. The language planner falls back to
a deterministic parser when no OpenVINO IR is present and **says so** in its status line — it
never pretends a model ran.

---

## How it works

### 1. Speech — Speechmatics realtime
`tandem/voice/` streams microphone or WAV audio to Speechmatics' realtime API and emits partial
and final transcripts with measured latency. Speaking while the arms are moving raises a
**barge-in**: the executor halts at the next safe point and re-plans. A stop vocabulary
(`stop`, `wait`, `abort`, …) is recognised explicitly.

With no `SPEECHMATICS_API_KEY`, the service switches to a scripted replay of the same events and
labels itself `REPLAY — not live ASR` everywhere it appears. It never fabricates a transcript.

### 2. Intent — a 1.5B model in OpenVINO INT4
`tandem/planner/` splits language understanding from planning, because a 1.5B model is good at
the first and bad at the second. Stage A asks the model for a small structured intent:

```json
{"place": ["plate", "fork", "spoon", "mug"], "pour": true, "refuse": null}
```

Stage B expands that intent into skills deterministically, against the **current** world state —
skipping what is already done, inserting `open_drawer` before anything that lives in the drawer,
and assigning arms by reachability. Because Stage B is a pure function of the world, re-running
it *is* re-planning, and the runner does exactly that after a step fails.

Backends: `openvino` (default, `Qwen2.5-1.5B-Instruct` INT4), `cloud` (OpenAI-compatible), and
`rules` (no model at all). Whichever ran is reported in `_meta` and on the dashboard.

### 3. The gate — deterministic, no model
`tandem/gate/` — described above. Roughly 200 lines of boring Python, and the only thing
standing between a confident language model and the actuators.

### 4. Skills — closed-loop, with honest failures
`tandem/control/` implements `open_drawer`, `pick`, `place`, `handoff`, `hold`, `pour`, `home`.
Each verifies its own post-condition against the simulator and returns a failure code rather
than raising. Notable pieces:

- **Damped-least-squares IK** restricted to each arm's five joints, with random restarts that
  are *biased towards the current configuration* — a restart that lands in a different elbow
  solution swings the gripper through a wide arc and throws whatever it was holding.
- **Tool-point targeting.** The wrist site is not the pinch point; it sits ~15 mm off along the
  jaw axis and the object seats several millimetres inside the fingers. Every pose is solved for
  where the object will actually be, and the seat depth is capped by how far the fingertips can
  go without driving through the table.
- **Close-on-contact grasping.** The jaws close until the servo's own tracking error says it is
  pressing, so an 8 % smaller plate under domain randomization is gripped just as firmly as a
  nominal one. No preset opening, no assumed object width.
- **Blocked-motion detection.** If the IK solved but the arm never arrived, that is reported as
  a failure instead of being discovered three steps later.

### 5. Perception and policy — learned, then quantized
`tandem/perception/` estimates object poses from the overhead and front cameras, replacing
privileged simulator state. `tandem/policy/` distills the scripted expert into an ACT-style
action-chunking policy over camera + proprioception. Both are exported to OpenVINO IR in FP32,
FP16 and NNCF INT8 and run through `openvino.CompiledModel` in the closed loop.

See [`PROOF.md`](PROOF.md) for what each one actually achieves, including where the policy is
worse than the expert it was distilled from.

### 6. Intel deployment
`scripts/benchmark_intel.py` reports the host, every OpenVINO device it can see, and
latency/throughput per precision per device, plus an accuracy-preservation table so the
quantization story is about *task quality*, not just megabytes.

⚠️ **Read the honesty note in `PROOF.md` before quoting any Intel number.** The machine this was
developed on is an AMD Ryzen with an NVIDIA discrete GPU. OpenVINO's CPU plugin runs there and
produces real IR, real INT8 and real measurements, but Intel does not validate that platform,
and there is no Intel iGPU or NPU on it. The benchmark script detects all of this and prints the
caveat itself; run it on a Core Ultra and it will light up CPU, GPU and NPU with no edits.

---

## Repository layout

```
assets/            MuJoCo scene, SO-101 meshes, demo audio
tandem/sim/        scene assembly, environment, seeded domain randomization
tandem/control/    IK, gripper calibration, skills, episode runner
tandem/gate/       the safety gate
tandem/planner/    intent parsing (OpenVINO / cloud / rules) and plan expansion
tandem/perception/ camera-based state estimation
tandem/policy/     distilled visuomotor policy
tandem/voice/      Speechmatics realtime and the scripted fallback
tandem/bench/      OpenVINO export, quantization and benchmarking
tandem/eval/       task and gate scorecards
tandem/server/     telemetry server for the dashboard
web/               landing page and live dashboard
scripts/           every command a reviewer needs
```

## Honest limits

Collected in one place rather than scattered: see [`MOCKS.md`](MOCKS.md) for the exact line
between what is real and what is simulated, and [`CLAIMS.md`](CLAIMS.md) for every claim in this
README tagged by the evidence behind it.

## License

MIT — see [`LICENSE`](LICENSE).
