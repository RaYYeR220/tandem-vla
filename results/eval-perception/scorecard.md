# Task scorecard — 6 randomized seeds

Instruction: **set the table for one and pour me some water**  
Seeds: `[0, 1, 2, 3, 4, 5]`  
Domain-randomization scale: `1.0`  
Planner: `deterministic-expansion`  
World state read from: **camera estimate**

## Task completion

- All six subgoals: **0/6** (0%)
- Place setting complete (plate, fork, spoon, mug): **0/6** (0%)
- Mean subgoal fraction: **0.14** (median 0.14, range 0.14–0.14)

## Per-subgoal

| subgoal | seeds passed | rate |
| --- | --- | --- |
| drawer_opened | 6/6 | 100% |
| plate_placed | 0/6 | 0% |
| fork_placed | 0/6 | 0% |
| spoon_placed | 0/6 | 0% |
| mug_placed | 0/6 | 0% |
| carton_emptied | 0/6 | 0% |
| water_in_cup | 0/6 | 0% |

## Per-skill reliability

Read the denominator carefully. `planned` counts every time the step appeared in a plan,
including after a re-plan, so a seed that keeps retrying one awkward grasp contributes
many rows. `gate blocked` are the ones the gate stopped before a joint moved — those
never reached the executor and are not counted against it. `rate` is over the steps that
actually executed.

| skill | planned | gate blocked | executed | succeeded | rate | first try |
| --- | --- | --- | --- | --- | --- | --- |
| handoff | 30 | 30 | 0 | 0 | 0% | 0% |
| hold | 6 | 6 | 0 | 0 | 0% | 0% |
| home | 12 | 0 | 12 | 12 | 100% | 100% |
| open_drawer | 14 | 0 | 14 | 12 | 86% | 64% |
| pick | 105 | 0 | 105 | 6 | 6% | 6% |
| place | 37 | 36 | 1 | 0 | 0% | 0% |
| pour | 6 | 6 | 0 | 0 | 0% | 0% |

## Failure modes, by count

| skill : code | count |
| --- | --- |
| `pick:GRIPPER_FULL` | 96 |
| `place:GRIPPER_EMPTY` | 36 |
| `handoff:GRIPPER_EMPTY` | 30 |
| `hold:GRIPPER_EMPTY` | 6 |
| `pour:GRIPPER_EMPTY` | 6 |
| `pick:OUT_OF_REACH` | 3 |
| `open_drawer:OUT_OF_REACH` | 2 |
| `place:OUT_OF_REACH` | 1 |

## Other

- Hand-off modes: `{}`
- Gate refusals raised: 78
- Re-plans triggered: 72
- Water landed in the mug: 0.0 units mean (of 10), 0.0 spilled
- Wall clock per episode: 34.0 s mean, 60.6 s max
