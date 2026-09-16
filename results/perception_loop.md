# Running the loop on camera estimates instead of privileged state

`scripts/evaluate.py --perception` swaps one constructor argument: the gate and the planner read
the world from `PerceptionEstimator` — two rendered camera views through an OpenVINO INT8 network —
instead of from `TandemEnv.world_state()`. Nothing else changes. Six seeds, same instruction, same
domain randomization.

| world state read from | mean subgoal fraction | place setting complete |
| --- | --- | --- |
| privileged simulator (20 seeds) | **0.64** | 6/20 |
| camera estimate (6 seeds) | **0.14** | 0/6 |

The swap works mechanically — the estimator returns the same schema, the planner expands it, the
gate judges it — and the task collapses anyway. That result is worth more than hiding it, because
the reason is specific and measurable.

## Why

Over 60 object observations across 12 seeds:

- position error: **median 35.5 mm**, p90 **417 mm**, max **623 mm**
- `reachable_by` disagrees with the truth on **18 of 60 objects (30%)**
- **53%** of objects sit within 40 mm of one of the reach-envelope boundaries

The median is fine. The tail is not: the network is usually within a few centimetres and
occasionally wrong by half a metre — typically on the cutlery, which is thin, metallic, and spends
most of its life inside a drawer or under an arm. A single object placed on the wrong side of the
0.280 m envelope makes the gate refuse a pick that would have succeeded, and with five objects per
episode the chance of getting all five right is small.

This is the gate behaving correctly on bad input, not the gate misbehaving. It refuses because the
scene it was handed says the object is out of reach. Garbage in, refusal out — which is the
failure mode you want, but it is still a failure.

## What would fix it

Not a better loss function. Two concrete things:

1. **A tail, not a median, target.** The training objective and the reported metric are both
   averages; what the gate needs is a bound. Training against p95 error, or predicting a
   per-object confidence and letting the gate treat low-confidence objects as "look again" rather
   than "out of reach", addresses the actual failure.
2. **Wrist cameras in the input.** The estimator sees only the overhead and front views, which is
   exactly the wrong pair for a thin object inside a drawer. The wrist cameras exist and are
   rendered; they are not wired into this network.

Neither was in scope for the time available, and neither is claimed to work.
