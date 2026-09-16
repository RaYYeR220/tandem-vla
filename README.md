# Tandem

**Two robot arms set a dinner table from a spoken instruction — and refuse the ones they can't ground.**

[Overview](https://rayyer220.github.io/tandem-vla/) · [Review in five minutes](JUDGES.md) · [Proof](PROOF.md) · [Honest limits](MOCKS.md)

Tandem is an end-to-end bimanual manipulation stack built on two simulated
[SO-101](https://github.com/TheRobotStudio/SO-ARM100) arms in MuJoCo. You say what you want.
A language model parses the intent. A deterministic gate checks every planned step against the
measured state of the scene and either allows it, repairs it, or refuses it with a reason. Then
the arms do it — handing objects to each other, because neither arm can reach both the props
and the place setting.

Language, perception and the visuomotor policy all run locally through **OpenVINO**. Speech is the
one exception and the one network call: transcription goes to Speechmatics' realtime API, and it
happens before the control loop starts, not inside it. Once an instruction has been heard, nothing
between the intent and the actuators touches the network.

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

pytest tandem/ -q                                     # 85 tests, ~13 s
python -m tandem.sim.scene                            # compile the cell, print its dimensions
python scripts/eval_gate.py                           # grade the safety gate (1 s)
python scripts/evaluate.py --seeds 20 --out results/eval   # the task scorecard
python scripts/benchmark_intel.py                     # OpenVINO device + latency report
python scripts/serve.py                               # the live dashboard on :8000

# The whole product in one command: speech in, cutlery on the table.
python scripts/voice_to_table.py --audio assets/audio/set_the_table.wav --seed 1

# The same instruction on ten randomized seeds, as one video.
python scripts/record_seed_montage.py --seeds 10 --out results/seeds.mp4
```

Nothing above needs an API key, a GPU, or a model download. The language planner falls back to
a deterministic parser when no OpenVINO IR is present and **says so** in its status line — it
never pretends a model ran.

---

## How it works

### 0. End to end
`scripts/voice_to_table.py` is the product in one command. It streams audio to Speechmatics,
takes the final transcript as the instruction, parses it locally through OpenVINO, expands it into
a plan against the current scene, gates every step, and runs it on the arms — printing at each
stage which backend actually served it. A captured run is in `results/voice_episode_log.txt`.

### 1. Speech — Speechmatics realtime
`tandem/voice/` streams microphone or WAV audio to Speechmatics' realtime API and emits partial
and final transcripts with measured latency. Speaking while the arms are moving raises a
**barge-in**: the executor halts at the next safe point and re-plans. A stop vocabulary
(`stop`, `wait`, `abort`, …) is recognised explicitly.

A captured live session is in `PROOF.md` section 6 and `results/voice_transcript.jsonl`: 22
partials, 11 finals, median final latency 922 ms, four barge-ins with the stop vocabulary flagged.

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

### 5. Perception and policy — one landed, one didn't
`tandem/perception/` estimates object poses from the overhead and front cameras, replacing
privileged simulator state. A shared conv trunk runs on each view separately and a per-object
spatial-attention map gives an explicit image coordinate through a soft-argmax; because the
cameras are rigid, the true positions can be projected analytically and used to supervise that
attention directly, which roughly halved the error against letting it emerge from the position
loss alone. Median position error **23.8 mm** over held-out seeds — 13–26 mm on the plate, mug
and carton, **48–56 mm with a 340 mm p90 on the cutlery**, which is thin, metallic and usually
inside a drawer. Drawer open-fraction is good to 2.2 mm.

`tandem/policy/` distills the scripted expert into an ACT-style action-chunking policy. **It does
not work.** Zero successful picks out of eighteen matched trials against the expert's fourteen,
and 1.0 of 7 subgoals end to end against the expert's 4.17. The cause is diagnosed rather than
guessed and is in the code: sixteen consecutive 50 Hz commands span a third of a second, over
which a servo target barely moves, so the L1 objective is minimised by a policy that echoes its
own proprioception and commits to no motion — val L1 under one degree while the arm sits still.
Re-striding the chunk to cover about a second halved the residual distance to the object and
still did not close a grasp. What remains is ordinary behaviour-cloning covariate shift.

Both nets export to OpenVINO IR at FP32, FP16 and NNCF INT8, calibrated on real recorded
observations rather than noise.

`scripts/evaluate.py --perception` swaps the world source from the simulator to the estimator with
one constructor argument and runs the whole task on camera input. The gate reaches the same verdict
as it would from privileged state on **94.1% of 456 gated steps** — but the *whole plan* comes out
identical only 58% of the time, because a nineteen-step plan need differ once, and the task score
drops to 0.14 against 0.64. Every disagreement is the cutlery. Written up in
[`PROOF.md`](PROOF.md) section 5b with the two things that would fix it.

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

## Ten seeds, one instruction

![Ten randomized seeds](results/seeds.mp4)

`results/seeds.mp4` plays the same instruction on ten randomized seeds at once — different object
positions and yaws, masses, frictions, sizes, lighting and materials in every tile, the same
sentence in all of them. Each tile carries its seed and its subgoal tally. Rebuild it with
`python scripts/record_seed_montage.py --seeds 10`.

## Results

<!-- RESULTS:START -->
**Safety gate — 24/24 cases.** 12/12 refused correctly, 4/4 repaired rather than refused, 8/8 legitimate steps allowed through — the negative control, without which the refusal score would mean nothing.

**Task — 20 randomized seeds, mean subgoal fraction 0.64.**

| subgoal | rate |
| --- | --- |
| drawer_opened | 20/20 (100%) |
| plate_placed | 18/20 (90%) |
| fork_placed | 18/20 (90%) |
| spoon_placed | 13/20 (65%) |
| mug_placed | 11/20 (55%) |
| carton_emptied | 10/20 (50%) |
| water_in_cup | 0/20 (0%) |

Hand-offs performed: `{'relay': 86}`. Gate refusals raised during execution: 352. Re-plans: 200.

**Quantization.** Perception and the distilled policy are exported to OpenVINO IR in FP32, FP16 and NNCF INT8, with the accuracy cost of each measured on held-out seeds rather than assumed — see `PROOF.md` section 4.
<!-- RESULTS:END -->

Full tables, with the failure modes and the per-skill breakdown, in [`PROOF.md`](PROOF.md).

## Honest limits

Collected in one place rather than scattered: see [`MOCKS.md`](MOCKS.md) for the exact line
between what is real and what is simulated, and [`CLAIMS.md`](CLAIMS.md) for every claim in this
README tagged by the evidence behind it.

## License

MIT — see [`LICENSE`](LICENSE).
