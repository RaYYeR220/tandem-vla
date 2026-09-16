# Scripted oracle vs distilled policy

Seeds 400..405 (6 of them), domain randomization scale 1.0. None of these seeds appear in the demonstration set (seeds 0..127).

## Matched skill trials (identical start state per trial)

| precision | controller | skill | attempts | successes | rate | mean s | median place err (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n/a | oracle | pick | 18 | 14 | 0.78 | 1.51 | - |
| fp32 | policy | pick | 18 | 0 | 0.00 | 2.73 | - |
| n/a | oracle | place | 11 | 11 | 1.00 | 1.35 | 2.2 |
| fp32 | policy | place | 11 | 0 | 0.00 | 2.93 | 106.5 |
| n/a | oracle | pick | 18 | 14 | 0.78 | 1.51 | - |
| int8 | policy | pick | 18 | 0 | 0.00 | 2.29 | - |
| n/a | oracle | place | 11 | 11 | 1.00 | 1.35 | 2.2 |
| int8 | policy | place | 11 | 1 | 0.09 | 2.24 | 137.6 |

## End-to-end episodes (full plan, gate and replanner active)

| precision | controller | episodes | mean subgoals | full success | pick rate | place rate | mean wall s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n/a | oracle | 6 | 4.167 / 7 | 0.00 | 0.323 | 0.356 | 101.7 |
| fp32 | policy | 6 | 1.0 / 7 | 0.00 | 0.0 | 0.0 | 93.4 |
| n/a | oracle | 6 | 4.167 / 7 | 0.00 | 0.323 | 0.356 | 101.7 |
| int8 | policy | 6 | 1.167 / 7 | 0.00 | 0.011 | 0.0 | 108.9 |
