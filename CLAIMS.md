# Claims ledger

Every public statement this project makes, tagged by the evidence behind it. The tiers:

| tier | meaning |
| --- | --- |
| **REPRODUCIBLE** | a command in this repo regenerates the number on any machine |
| **VERIFIED-LIVE** | observed running against a real external service or real hardware |
| **MEASURED-HERE** | measured on the development machine; reproducible, but the number is host-specific |
| **NOT-CLAIMED** | something a reader might reasonably assume, that we are explicitly *not* asserting |

---

## Architecture and behaviour

| Claim | Tier | Evidence |
| --- | --- | --- |
| The safety gate refuses ungrounded instructions and repairs wrong arm assignments | **REPRODUCIBLE** | `python scripts/eval_gate.py` — scorecard in `PROOF.md §1`, sources in `tandem/gate/` |
| The gate's refusal score is not gamed by refusing everything | **REPRODUCIBLE** | eight negative-control ALLOW cases in the same suite, scored separately |
| No language model participates in the safety decision | **REPRODUCIBLE** | `tandem/gate/__init__.py` imports only `numpy` and `tandem.sim.layout` |
| A hand-off is structurally required, not decorative | **REPRODUCIBLE** | `pytest tandem/sim/test_sim.py` asserts across 12 seeds that no prop is ever reachable by the right arm and no slot by the left; `scripts/eval_gate.py` shows the gate inserting the hand-off |
| The whole suite passes | **REPRODUCIBLE** | `pytest tandem/ -q` — 85 tests covering the cell's shape, seed determinism, the territory invariant, gripper calibration, IK, the planner, the voice layer and the OpenVINO bench |
| Grasps are decided by contact, not by proximity or attachment | **REPRODUCIBLE** | `TandemEnv.jaw_contacts` reads `data.contact`; `tandem/control/primitives.py::close_on_object` closes until the servo's tracking error reports load |
| Episodes are reproducible from a single integer seed | **REPRODUCIBLE** | `tandem/sim/randomize.py`; re-running `scripts/evaluate.py` with the same seeds reproduces the flags |
| Re-planning is state-derived, not scripted | **REPRODUCIBLE** | `tandem/eval/tasks.py::canonical_plan` is a pure function of the world state; the runner calls it again after a failure |
| Task success rates across randomized seeds | **REPRODUCIBLE** | `python scripts/evaluate.py --seeds 20` — `PROOF.md` §2, including per-skill rates and the failure-mode histogram. Run twice end to end: one subgoal out of 140 moved between runs (`PROOF.md` §2b), and both runs are kept. |

## Intel and OpenVINO

| Claim | Tier | Evidence |
| --- | --- | --- |
| Models are exported to genuine OpenVINO IR and quantized with NNCF | **REPRODUCIBLE** | `tandem/bench/ovutil.py`, `scripts/export_models.py`; precision is detected by reading constant element types out of the IR, not from filenames |
| Latency, throughput and model size per precision per device | **MEASURED-HERE** | `python scripts/benchmark_intel.py` — `PROOF.md §3` |
| Quantization preserves task quality | **REPRODUCIBLE** | accuracy-preservation table in `PROOF.md` §4. FP32 to FP16 is free to three digits; INT8 costs 10 mm of median perception error and leaves the p90 tail untouched. |
| Perception INT8 is faster and smaller on CPU | **MEASURED-HERE** | 5.02 ms p50 against 7.20 ms FP32, 1.73 MiB against 6.26 MiB. A quieter earlier measurement of the same pair read 2.61 vs 5.80 ms, so the honest range is 1.4x to 2.0x depending on machine load. |
| Policy INT8 is a regression on this CPU | **MEASURED-HERE** | 6.13 ms p50 against FP32's 3.00 ms, reproduced across two runs, and roughly double the action error. The policy would ship at FP16. Published because a quantization section containing only wins is not a measurement. |
| The gate reaches the same verdict from camera input as from privileged state | **REPRODUCIBLE** | 0.941 per-step agreement over 456 gated steps on 24 held-out seeds (`PROOF.md` §4). The *whole plan* comes out identical only 58% of the time, and that weaker number is published beside it. |
| The distilled policy can drive the task | **NOT-CLAIMED** | It cannot. 0 of 18 matched picks against the expert's 14, 1.0 of 7 subgoals end to end against 4.17. Diagnosed, partially fixed, unsolved — `PROOF.md` §7. |
| The benchmark runs on Intel Core Ultra Series 2/3 and reports CPU, iGPU and NPU | **NOT-CLAIMED** | We have no Intel silicon. The script is written to enumerate whatever OpenVINO exposes and prints its own caveat when the host is not Intel; it has never been run on a Core Ultra by us. |
| Any latency figure here characterises Intel hardware | **NOT-CLAIMED** | The host is an AMD Ryzen 5 5600X with an NVIDIA discrete GPU. Both are named in every report. |

## Language

| Claim | Tier | Evidence |
| --- | --- | --- |
| Intent parsing runs locally on a 1.5B model through OpenVINO INT4 | **MEASURED-HERE** | `python tandem/planner/benchmark.py` — accuracy and latency for the 1.5B and 0.5B exports, `PROOF.md §4` |
| The planner reports which backend actually ran | **REPRODUCIBLE** | `tandem/planner/backends.py::info()`; `_meta.backend` and `_meta.fell_back_from` in every plan |
| Refusal of absent objects is decided in code, not by the model | **REPRODUCIBLE** | `scene_guard()` in `tandem/planner/planner.py`. The raw model columns in the planner scorecard show what the models manage unaided, which is 1/4 and 0/4. We publish both. |
| The 1.5B model can produce a correct multi-step plan unaided | **NOT-CLAIMED** | It cannot. That is why planning is a deterministic expansion and the model only parses intent. The measurement that led to this design is in `PROOF.md §4`. |

## Voice

| Claim | Tier | Evidence |
| --- | --- | --- |
| Realtime transcription uses the current Speechmatics SDK against the live endpoint | **VERIFIED-LIVE** | `speechmatics-rt` 1.1.1 against `wss://eu2.rt.speechmatics.com/v2`. Full captured session in `results/voice_transcript.jsonl` and `PROOF.md` section 6: 22 partials, 11 finals, median final latency 922 ms. The authentication-failure path was exercised separately and produces one clean error event. |
| Barge-in fires on live speech, not just in tests | **VERIFIED-LIVE** | four `barge_in` events in the captured session, the first flagged `stop` from the stop vocabulary |
| Barge-in halts execution and triggers a re-plan | **REPRODUCIBLE** | `tandem/voice/test_voice.py`, 8 passing tests |
| The system never fabricates a transcript | **REPRODUCIBLE** | with no key, `ScriptedTranscriber` reports `{"backend": "scripted", "live": false}` and the UI is labelled REPLAY |
| Reported transcription latency is a lab-grade measurement | **NOT-CLAIMED** | It is an approximation from chunk-send time to final arrival, documented as such in `tandem/voice/speechmatics_rt.py`. It slightly overstates true latency and cannot see network buffering. |

| Nothing in the system touches the network | **NOT-CLAIMED** | Transcription is a call to Speechmatics' realtime API. It happens before the control loop and is the only network call, but it is a network call, and the README says so. Language, perception and the policy do run locally through OpenVINO. |

## Things we are explicitly not claiming

- **That the arms hand objects to each other in mid-air.** They do not, by default. See `MOCKS.md`.
- **That water lands in the cup.** It does not. The carton empties, both arms coordinate, and the
  stream misses. Scored as two separate subgoals so the working half is not credited for the
  broken half.
- **That this ran on physical SO-101 hardware.** It did not. The challenge is simulation-first and
  no hardware was involved at any point.
- **That the water is a fluid simulation.** It is ten rigid particles.
- **That the perception or policy networks match the scripted expert.** Whatever the head-to-head
  in `PROOF.md §5` says is what they do; if the policy is worse, that is what is printed.
