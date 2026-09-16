# Task scorecard — 20 randomized seeds

Instruction: **set the table for one and pour me some water**  
Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`  
Domain-randomization scale: `1.0`  
Planner: `deterministic-expansion`  
World state read from: **privileged simulator**

## Task completion

- All six subgoals: **0/20** (0%)
- Place setting complete (plate, fork, spoon, mug): **6/20** (30%)
- Mean subgoal fraction: **0.65** (median 0.71, range 0.29–0.86)

## Per-subgoal

| subgoal | seeds passed | rate |
| --- | --- | --- |
| drawer_opened | 20/20 | 100% |
| plate_placed | 18/20 | 90% |
| fork_placed | 18/20 | 90% |
| spoon_placed | 13/20 | 65% |
| mug_placed | 11/20 | 55% |
| carton_emptied | 11/20 | 55% |
| water_in_cup | 0/20 | 0% |

## Per-skill reliability

Read the denominator carefully. `planned` counts every time the step appeared in a plan,
including after a re-plan, so a seed that keeps retrying one awkward grasp contributes
many rows. `gate blocked` are the ones the gate stopped before a joint moved — those
never reached the executor and are not counted against it. `rate` is over the steps that
actually executed.

| skill | planned | gate blocked | executed | succeeded | rate | first try |
| --- | --- | --- | --- | --- | --- | --- |
| handoff | 159 | 73 | 86 | 74 | 86% | 86% |
| hold | 145 | 9 | 136 | 136 | 100% | 100% |
| home | 36 | 0 | 36 | 36 | 100% | 100% |
| open_drawer | 20 | 0 | 20 | 20 | 100% | 100% |
| pick | 397 | 206 | 191 | 101 | 53% | 48% |
| place | 162 | 75 | 87 | 67 | 77% | 77% |
| pour | 145 | 12 | 133 | 0 | 0% | 0% |

## Failure modes, by count

| skill : code | count |
| --- | --- |
| `pick:GRIPPER_FULL` | 196 |
| `pour:OUT_OF_REACH` | 82 |
| `pick:OUT_OF_REACH` | 82 |
| `place:GRIPPER_EMPTY` | 75 |
| `handoff:GRIPPER_EMPTY` | 73 |
| `pour:POUR_MISSED` | 51 |
| `pick:GRASP_FAILED` | 18 |
| `place:PLACE_MISSED` | 11 |
| `handoff:OUT_OF_REACH` | 10 |
| `place:OUT_OF_REACH` | 9 |
| `hold:GRIPPER_EMPTY` | 9 |
| `pour:GRIPPER_EMPTY` | 8 |
| `pour:NO_HOLDER_FOR_POUR` | 4 |
| `handoff:GRASP_FAILED` | 2 |

## Other

- Hand-off modes: `{'relay': 86}`
- Gate refusals raised: 375
- Re-plans triggered: 224
- Water landed in the mug: 0.0 units mean (of 10), 6.0 spilled
- Wall clock per episode: 108.6 s mean, 318.1 s max
