# Tandem — internal module contracts

Frozen interfaces every module codes against. Do not change a signature without updating
this file first.

## Vocabulary

Arms: `"left" | "right"`.

Objects: `"plate" | "mug" | "bottle" | "spoon" | "fork"` and the articulated `"drawer"`.

Slots (targets on the table): `"slot_plate" | "slot_fork" | "slot_spoon" | "slot_mug"`.

Skills (the only verbs a plan may contain):

| skill | args | meaning |
| --- | --- | --- |
| `open_drawer` | `{arm}` | pull the drawer fully open |
| `close_drawer` | `{arm}` | push the drawer shut |
| `pick` | `{arm, object}` | grasp and lift `object` |
| `place` | `{arm, object, target}` | put the held `object` down at `target` (a slot name) |
| `handoff` | `{from_arm, to_arm, object}` | transfer the held object in the shared zone |
| `hold` | `{arm, object}` | keep the held object still while the other arm acts |
| `pour` | `{arm, source, target}` | tilt `source` over `target` (the other arm must `hold` `target`) |
| `home` | `{arm}` | retract to the rest pose |

## World state (`tandem.sim.env.TandemEnv.world_state()` and `perception.estimate()`)

```jsonc
{
  "t": 12.34,                         // sim seconds
  "objects": {
    "plate": {
      "pos": [x, y, z],               // metres, table frame
      "yaw": 0.31,                    // radians
      "held_by": null,                // null | "left" | "right"
      "on_slot": null,                // null | slot name, if resting on its slot
      "in_drawer": false,
      "reachable_by": ["left"]        // subset of ["left","right"]
    }
    // ... one entry per object
  },
  "drawer": {"open_frac": 0.0, "is_open": false, "reachable_by": ["left"]},
  "slots": {"slot_plate": {"pos": [x,y,z], "occupied_by": null, "reachable_by": ["right"]}},
  "arms": {
    "left":  {"holding": null, "tcp": [x,y,z], "qpos": [5 floats], "gripper": 0.0, "busy": false}
  },
  "water": {"in_mug": 0, "spilled": 0, "in_bottle": 10}
}
```

Every field is derived from the *estimated* scene when perception is on, and from privileged
sim state when it is off. Both paths produce the identical schema.

## Plan (planner output, gate input)

```jsonc
{
  "instruction": "set the table for one and pour me some water",
  "goal_summary": "plate, fork, spoon, mug at the place setting; water in the mug",
  "steps": [
    {"id": 1, "skill": "open_drawer", "args": {"arm": "left"},
     "rationale": "the cutlery is inside the drawer"}
  ]
}
```

Rules the planner must honour (the gate re-checks all of them):

- ids are 1-based and contiguous.
- every `args` key must match the skill's arg list exactly.
- `pick` requires the arm to be empty; `place`/`handoff`/`pour` require it to be holding.
- an object may only be `pick`ed by an arm that can reach it.

## Gate verdict (`tandem.gate.check_step`)

```jsonc
{
  "step_id": 3,
  "verdict": "ALLOW" | "REFUSE" | "REWRITE",
  "code": "OK" | "OUT_OF_REACH" | "GRIPPER_FULL" | "GRIPPER_EMPTY" | "OBJECT_ABSENT" |
          "OBJECT_OCCLUDED" | "SLOT_OCCUPIED" | "DRAWER_CLOSED" | "SELF_COLLISION" |
          "SHARED_ZONE_BUSY" | "NO_HOLDER_FOR_POUR" | "UNKNOWN_OBJECT" | "UNKNOWN_SKILL",
  "reason": "the mug is 0.32 m from the right base, past the 0.28 m envelope",
  "rewrite": [ /* replacement steps, only when verdict == "REWRITE" */ ]
}
```

`REWRITE` is how a hand-off gets inserted: a `place`/`pick` the proposing arm cannot reach is
rewritten into `handoff` + `place` by the other arm. `REFUSE` is terminal for that step and is
surfaced to the user verbatim.

## Voice events (`tandem.voice`)

The voice service is a producer of events on an `asyncio.Queue`:

```jsonc
{"type": "partial", "text": "put the plate on the", "t": 1.9}
{"type": "final",   "text": "put the plate on the table", "t": 2.4, "latency_ms": 310}
{"type": "barge_in","text": "stop",                        "t": 5.1}
{"type": "error",   "detail": "..."}
```

`barge_in` is emitted when a final transcript arrives while the executor is running; the
executor treats it as a request to halt at the next safe point and replan.

## Telemetry stream (backend -> dashboard, newline-delimited JSON over SSE)

```jsonc
{"type":"transcript","stage":"partial|final","text":"..."}
{"type":"plan","plan":{...}}
{"type":"verdict","verdict":{...}}
{"type":"step","step_id":3,"status":"running|done|failed","skill":"pick","arm":"left"}
{"type":"state","world":{...}}
{"type":"frame","cam":"overhead","jpeg_b64":"..."}
{"type":"infer","model":"policy|perception|planner","device":"CPU","ms":4.2,"precision":"INT8"}
{"type":"episode","seed":7,"status":"success|failure","score":{...}}
```
