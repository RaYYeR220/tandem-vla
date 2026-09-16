# Precision vs task quality

Host: `AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD` -- Intel-validated: **False**

> Host CPU is 'AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD', not an Intel part. This run is NOT on Intel-validated hardware (no Core Ultra Series 2/3 here). OpenVINO's CPU/GPU plugins still execute and the numbers below are real measurements, but they characterize whatever silicon actually ran them, not Intel's target hardware. Any 'GPU' or 'NPU' device listed is reported by FULL_DEVICE_NAME below so it can't be mistaken for an Intel iGPU/NPU.

All numbers on held-out seeds neither network trained on. INT8 is NNCF post-training quantization calibrated on recorded observations from the same datasets.

## Perception (600 held-out frames)

| variant | IR precision | size (MiB) | median err (mm) | p90 err (mm) | drawer MAE (mm) | plate median | mug median | bottle median | spoon median | fork median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | FP32 | nan | 23.8 | 236.7 | 2.20 | 29.6 | 14.0 | 13.8 | 58.2 | 54.1 |
| OpenVINO FP32 | FP32 | 6.26 | 23.8 | 236.7 | 2.20 | 29.6 | 14.0 | 13.8 | 58.2 | 54.1 |
| OpenVINO FP16 | FP16 | 3.19 | 23.8 | 236.2 | 2.20 | 29.6 | 14.0 | 13.8 | 58.2 | 54.4 |
| OpenVINO INT8 | INT8 | 1.73 | 34.0 | 236.9 | 2.37 | 42.1 | 17.3 | 17.0 | 63.7 | 57.7 |

Live drop-in check: the environment is reset on 24 held-out seeds, the two cameras are rendered from it, and the estimator builds a whole world-state dict from those frames plus the arms' own joint encoders. Agreement is against `TandemEnv.world_state()` on the same instant.

| variant | schema match | reachable_by | on_slot | in_drawer | held_by | median err (mm) | drawer MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenVINO FP32 | True | 0.817 | 0.983 | 0.792 | 1.000 | 25.7 | 0.0153 |
| OpenVINO FP16 | True | 0.817 | 0.983 | 0.792 | 1.000 | 25.7 | 0.0153 |
| OpenVINO INT8 | True | 0.817 | 0.975 | 0.783 | 1.000 | 29.0 | 0.0185 |

**The decision the stack actually makes.** Both world states -- one from the cameras, one privileged -- are pushed through `tandem.eval.tasks.canonical_plan` and then through `tandem.gate.check_step` for every step of the privileged plan, so a verdict difference is attributable to the scene estimate and nothing else.

| variant | plan identical | verdict sequence identical | per-step verdict agreement | gated steps |
| --- | --- | --- | --- | --- |
| OpenVINO FP32 | 0.58 | 0.21 | 0.941 | 456 |
| OpenVINO FP16 | 0.58 | 0.21 | 0.941 | 456 |
| OpenVINO INT8 | 0.54 | 0.21 | 0.939 | 456 |

Where INT8 disagreed with privileged state, by step:

- 17x `pick: REWRITE/DRAWER_CLOSED -> ALLOW/OK`
- 8x `pick: REWRITE/DRAWER_CLOSED -> REWRITE/OUT_OF_REACH`
- 2x `pick: ALLOW/OK -> REWRITE/OUT_OF_REACH`
- 1x `pick: ALLOW/OK -> REWRITE/DRAWER_CLOSED`

Symbolic agreement -- how often the booleans the planner and gate actually read come out the same from the estimate as from privileged state, over every (frame, object) pair:

| variant | reachable_by | on_slot | in_drawer | drawer.is_open |
| --- | --- | --- | --- | --- |
| PyTorch FP32 | 0.783 | 0.909 | 0.957 | 0.978 |
| OpenVINO FP32 | 0.783 | 0.909 | 0.957 | 0.978 |
| OpenVINO FP16 | 0.783 | 0.908 | 0.957 | 0.978 |
| OpenVINO INT8 | 0.763 | 0.894 | 0.953 | 0.980 |

## Policy (600 held-out transitions)

| variant | IR precision | size (MiB) | action MSE | action L1 | first-step MSE |
| --- | --- | --- | --- | --- | --- |
| PyTorch FP32 | FP32 | nan | 1.048e-03 | 0.00942 | 2.939e-04 |
| OpenVINO FP32 | FP32 | 26.43 | 1.048e-03 | 0.00942 | 2.939e-04 |
| OpenVINO FP16 | FP16 | 13.44 | 1.048e-03 | 0.00943 | 2.941e-04 |
| OpenVINO INT8 | INT8 | 8.14 | 2.029e-03 | 0.02625 | 1.142e-03 |

Closed-loop task success at FP32 vs INT8 is measured separately by `scripts/eval_policy.py --precision fp32,int8`, which runs the same held-out seeds through the simulator under each IR.
