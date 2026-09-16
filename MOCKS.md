# What is real, and what is simulated

One page, no hedging. If something in this project is not what it looks like, it is listed here.

## Simulated by design

| Thing | Status |
| --- | --- |
| The robot | **Simulated.** Two SO-101 arms in MuJoCo 3.13, using the official `robotstudio_so101` model from `mujoco_menagerie` — real link geometry, real joint limits, real Feetech STS3215 servo parameters. No physical hardware was involved. The challenge this was built for is simulation-first. |
| The table, drawer and props | **Authored by us**, in `assets/scene_table.xml`. Dimensions are chosen to fit the SO-101's measured gripper envelope, and `tandem/sim/layout.py` records why each number is what it is. |
| Water | **Ten rigid particles**, not a fluid. Pouring is scored by counting how many end up inside the cup's walls — real rigid-body contact, but not computational fluid dynamics. |
| Cameras | MuJoCo's renderer. Real projection and real occlusion; no sensor noise model, no rolling shutter, no lens distortion. |

## Real, and verifiable

| Thing | Status |
| --- | --- |
| Physics | Real MuJoCo contact dynamics at 2 ms, elliptic friction cone. Grasps succeed or fail by contact forces, not by a scripted attach. Nothing is ever teleported into the gripper. |
| Grasp detection | Read from `data.contact` — actual jaw-to-object contacts — not from proximity or from an assumption that the close command worked. |
| The safety gate | Real deterministic code (`tandem/gate/`), graded by `scripts/eval_gate.py` against a case suite with a negative control. No model involved. |
| Domain randomization | Real per-seed perturbation of position, yaw, mass, friction, size, lighting, materials and drawer friction. One integer seed reproduces an episode exactly. |
| OpenVINO | Real `ov.convert_model` exports, real NNCF post-training INT8, real `ov.CompiledModel` inference in the closed loop. The IR files are rebuilt by `scripts/export_models.py`; nothing is faked with a timer. |
| Speechmatics | Real realtime WebSocket API when `SPEECHMATICS_API_KEY` is set. The error path was verified end-to-end against the live endpoint with a deliberately invalid key. |
| The language planner | A real `Qwen2.5-1.5B-Instruct` INT4 OpenVINO pipeline when the IR is present. Latency and accuracy are measured, not estimated. |

## Substitutions, and when they are used

| Substitution | When it engages | How you can tell |
| --- | --- | --- |
| Scripted transcript replay instead of Speechmatics | No `SPEECHMATICS_API_KEY` | `VoiceService.info()` returns `{"backend": "scripted", "live": false}` and the dashboard shows **REPLAY — not live ASR**. It never emits a transcript it did not replay. |
| Keyword intent parser instead of the language model | No OpenVINO IR in `models/planner-ov`, or `TANDEM_PLANNER=rules` | `_meta.backend` reads `rules`, and the dashboard HUD says `rules`. The model is never credited for a parse it did not do. |
| Shared-zone relay instead of an in-air hand-off | Always, unless `TANDEM_HANDOFF=direct` | The step result carries `"mode": "relay"`, and the scorecard reports hand-off modes separately. See *Honest limits* below. |
| Scripted expert instead of the distilled policy | The default execution path | The policy is evaluated head-to-head against the expert in `PROOF.md`; whichever one ran is named in the results. |

**Speech is a cloud call.** Everything else neural runs on the machine through OpenVINO, but
transcription goes to Speechmatics. It happens before the control loop and nothing between the
parsed intent and the actuators leaves the box — but "fully on-device" would be false and we do
not say it.

## Honest limits

**The hand-off is a relay, not an in-air exchange.** The props start in the left arm's territory
and the place setting is in the right arm's, so objects genuinely have to change arms — that part
is structural. But with the two bases 0.40 m apart and both wrists forced near-vertical by the
workspace, the arms' own links collide above the exchange point before the jaws ever meet. We
measured this, kept the direct path behind `TANDEM_HANDOFF=direct` for elongated props, and made
the shared-zone relay the default. The object still changes arms; it touches down in the middle
on the way.

**The pour empties the carton but misses the cup.** Both arms coordinate correctly — one presents
the cup, the other brings the carton over it and tips past 100° — and the contents do leave the
carton. They land roughly 8 cm away from a 37 mm opening. The cause is measured, not guessed: the
carton rotates inside a ~12 N pinch during the tilt, so the achieved rim pose diverges from the
solved one. Three fixes were tried and measured, and all three failed: pre-aiming the cup at the
rim predicted from the solved pose, re-solving the carton pose from the observed error, and
tipping in nine stages with the cup tracking the spout between each. The residual stays around
75–85 mm. This is why the scorecard scores `carton_emptied` and `water_in_cup` as two separate
subgoals rather than one flag: the coordination works, the marksmanship does not.

The honest read is that a reliable pour needs the carton held rigidly rather than pinched — a
wrist-mounted fixture, or a grasp that constrains rotation about the jaw axis. That is a hardware
answer to a hardware problem, and we did not have one.

**The camera-based estimator is a drop-in, but it does not yet carry the task.** Running the
evaluation with `--perception` swaps one constructor argument and the gate and planner read the
scene from two rendered views through an OpenVINO INT8 network instead of from privileged state.
It works mechanically and the task collapses: 0.14 mean subgoal fraction against 0.64. The reason
is a tail, not a bias — median position error 35 mm but p90 of 417 mm, and 30% of objects end up
on the wrong side of the 280 mm reach boundary, so the gate correctly refuses picks that would
have succeeded. Full write-up and the two things that would fix it are in `PROOF.md` section 5b.
The default evaluation therefore runs on privileged state, and says so in its own header.

**The distilled policy does not work.** Zero successful picks in eighteen matched trials against
the scripted expert's fourteen; 1.0 of 7 subgoals end to end against 4.17. The failure is
understood — an action chunk covering a third of a second is minimised by predicting no motion,
and re-striding it to about a second halved the residual distance without closing a grasp — but
understood is not fixed. Every execution number elsewhere in this repo is the scripted expert
unless it says otherwise, and the head-to-head is in `PROOF.md` section 7 rather than omitted.

**Perception cannot see the water, and says so.** The particles are inside an opaque carton and
inside a mug viewed from above. The estimator returns zeros for them and sets
`perception.water_observed = False` rather than inventing a count.

**Yaw is unlearnable for the round props.** Plate, mug and carton are rotationally symmetric, so
their yaw label carries no information and the network scores about 90 degrees, which is chance.
The elongated cutlery comes in at 14 degrees. Reported rather than quietly dropped from the table.

**The perception dataset predates the current place-setting layout.** Slot positions moved after
the 18k frames were collected. Nothing in the estimator hardcodes them — it reads `layout.SLOTS`
live — but the training scenes are from the older arrangement, and the same checkpoint measures
15.9 mm median on the old layout against 25.7 mm on the current one. Re-collecting and retraining
is about 35 minutes and is the cheapest improvement available; it was not done, and the worse
number is the one published.

**OpenVINO's GPU plugin cannot compile the policy transformer** on this machine's NVIDIA card
(`CL_BUILD_PROGRAM_FAILURE`). It is recorded as a skipped row with the reason in the benchmark CSV
rather than dropped from the table.

**The gripper envelope is smaller than it looks.** The SO-101's fingertips travel 133 mm apart,
but the jaw faces close in well ahead of them; the real usable opening is about 50 mm. Every prop
is sized to fit inside that. A wider plate is not a harder version of this task — it is an
ungraspable one for this arm.

**No Intel silicon.** See the caveat at the top of `PROOF.md`.
