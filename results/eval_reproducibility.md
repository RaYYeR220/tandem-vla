# Two independent runs of the same twenty seeds

The evaluation was run twice, end to end, on the same seeds and the same code. Episodes are not
bit-identical between runs: the runner has a wall-clock budget, so a machine under different load
gives a step a different number of attempts before the budget stops it. That is a real property of
the harness and it is worth measuring rather than assuming away.

| subgoal | run A | run B |
| --- | --- | --- |
| drawer_opened | 20/20 | 20/20 |
| plate_placed | 18/20 | 18/20 |
| fork_placed | 18/20 | 18/20 |
| spoon_placed | 13/20 | 13/20 |
| mug_placed | 11/20 | 11/20 |
| carton_emptied | 10/20 | **11/20** |
| water_in_cup | 0/20 | 0/20 |
| **mean subgoal fraction** | **0.643** | **0.650** |

One subgoal out of 140 moved. The numbers published elsewhere in this repo and quoted in the demo
video are run B; run A is kept at `results/eval-run-a/` so the comparison can be checked rather
than taken on trust.

The domain randomization itself *is* deterministic — the same seed always produces the same scene,
and `pytest tandem/sim/test_sim.py` asserts it. The variance above is in execution, not in setup.
