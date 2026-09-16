"""Planner tests. Run with pytest, or directly:

    python tandem/planner/test_planner.py

Nothing here touches a model or the network. Stage A is covered with the keyword parser
and with stub backends that reproduce the exact slop real models emit; stage B is
deterministic, so it is asserted on directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tandem.planner.backends import RuleBasedPlanner, get_planner
from tandem.planner.expander import ExpansionError, expand
from tandem.planner.planner import (
    extract_json,
    parse_intent,
    plan,
    replan,
    salvage_intent_fields,
    scene_guard,
)
from tandem.planner.prompt import build_intent_prompt, summarize_world
from tandem.planner.schema import PLACEABLE, STAGING, normalize_intent, repair_plan, validate_plan


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def _fake_world() -> dict:
    """A reset workcell: props in left-arm territory, place setting in right-arm territory."""
    return {
        "t": 0.0,
        "objects": {
            "plate": {
                "pos": [-0.16, 0.16, 0.006], "yaw": 0.0, "held_by": None,
                "on_slot": None, "in_drawer": False, "reachable_by": ["left"],
            },
            "mug": {
                "pos": [-0.20, 0.10, 0.031], "yaw": 0.0, "held_by": None,
                "on_slot": None, "in_drawer": False, "reachable_by": ["left"],
            },
            "bottle": {
                "pos": [-0.23, 0.13, 0.051], "yaw": 0.0, "held_by": None,
                "on_slot": None, "in_drawer": False, "reachable_by": ["left"],
            },
            "spoon": {
                "pos": [-0.34, 0.18, 0.02], "yaw": 0.0, "held_by": None,
                "on_slot": None, "in_drawer": True, "reachable_by": ["left"],
            },
            "fork": {
                "pos": [-0.34, 0.20, 0.02], "yaw": 0.0, "held_by": None,
                "on_slot": None, "in_drawer": True, "reachable_by": ["left"],
            },
        },
        "drawer": {"open_frac": 0.0, "is_open": False, "reachable_by": ["left"]},
        "slots": {
            "slot_plate": {"pos": [0.27, 0.14, 0.006], "occupied_by": None, "reachable_by": ["right"]},
            "slot_fork": {"pos": [0.195, 0.14, 0.006], "occupied_by": None, "reachable_by": ["right"]},
            "slot_spoon": {"pos": [0.345, 0.14, 0.006], "occupied_by": None, "reachable_by": ["right"]},
            "slot_mug": {"pos": [0.305, 0.215, 0.032], "occupied_by": None, "reachable_by": ["right"]},
        },
        "arms": {
            "left": {"holding": None, "tcp": [-0.2, 0.05, 0.12], "qpos": [0.0] * 5, "gripper": 0.0, "busy": False},
            "right": {"holding": None, "tcp": [0.2, 0.05, 0.12], "qpos": [0.0] * 5, "gripper": 0.0, "busy": False},
        },
        "water": {"in_mug": 0, "spilled": 0, "in_bottle": 10},
    }


def _intent(**overrides) -> dict:
    return normalize_intent({"place": [], "pour": False, **overrides})


def _step(skill: str, **args) -> dict:
    return {"id": 0, "skill": skill, "args": args, "rationale": ""}


def _numbered(*steps: dict) -> dict:
    out = []
    for index, step in enumerate(steps):
        step = dict(step)
        step["id"] = index + 1
        out.append(step)
    return {"steps": out}


def _skills(result: dict) -> list[str]:
    return [s["skill"] for s in result["steps"]]


class _Stub:
    """A stage-A backend that returns a fixed string."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.last_tokens_per_s = 0.0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.text

    def info(self) -> dict:
        return {"backend": "stub", "model": "stub", "device": "none", "precision": "n/a"}


# --------------------------------------------------------------------------------------
# stage A — the language model's only job
# --------------------------------------------------------------------------------------


def test_rules_parse_intent_full_setting():
    intent = RuleBasedPlanner().parse_intent("set the table for one")
    assert intent["place"] == list(PLACEABLE)
    assert intent["pour"] is False
    assert intent["refuse"] is None


def test_rules_parse_intent_partial_request():
    intent = RuleBasedPlanner().parse_intent("just put the plate and the fork out")
    assert intent["place"] == ["plate", "fork"]
    assert intent["pour"] is False


def test_rules_parse_intent_pour_owns_the_mug():
    intent = RuleBasedPlanner().parse_intent("fill my mug please")
    assert intent["pour"] is True
    assert intent["place"] == [], "the pour leg fetches and returns the mug itself"


def test_rules_parse_intent_drawer_and_refusal():
    rules = RuleBasedPlanner()
    assert rules.parse_intent("open the drawer")["open_drawer"] is True
    assert rules.parse_intent("close the drawer")["close_drawer"] is True
    refused = rules.parse_intent("cut the bread with a knife")
    assert refused["refuse"] and "knife" in refused["refuse"]
    assert refused["place"] == [] and refused["pour"] is False


def test_parse_intent_reports_the_backend_and_cost():
    intent, meta = parse_intent("set the table for one", backend="rules")
    assert intent["place"] == list(PLACEABLE)
    assert meta["backend"] == "rules"
    assert meta["stage_a_ms"] >= 0.0
    assert meta["attempts"] == 1


def test_normalize_intent_absorbs_model_slop():
    intent = normalize_intent({
        "place": "a cup and a dish", "pour": "yes",
        "open_drawer": "null", "close_drawer": "false", "refuse": "none",
    })
    assert intent["place"] == ["plate", "mug"], "canonical setting order, aliases resolved"
    assert intent["pour"] is True
    assert intent["open_drawer"] is None
    assert intent["close_drawer"] is False
    assert intent["refuse"] is None


def test_normalize_intent_never_invents_an_object():
    assert normalize_intent({"place": ["knife", "napkin"]})["place"] == []


def test_normalize_intent_ignores_a_degenerate_refusal():
    """Small models fill the refuse box with 'nothing to refuse'. That is not a refusal."""
    for junk in ("nothing to refuse", "none", "No refusal needed", "n/a", "-"):
        assert normalize_intent({"place": ["plate"], "refuse": junk})["refuse"] is None
    assert normalize_intent({"refuse": "there is no knife"})["refuse"] == "there is no knife"


def test_scene_guard_overrules_a_helpful_model():
    """The model would rather fetch a knife than admit there is none. Code decides."""
    helpful = _Stub('{"place": ["plate"], "pour": false, "refuse": null, "paraphrase": "get the knife"}')

    guarded, _ = parse_intent("get me a knife", backend=helpful)
    assert guarded["refuse"] == "there is no knife in the scene"
    assert guarded["place"] == [] and guarded["pour"] is False

    raw, _ = parse_intent("get me a knife", backend=helpful, guard=False)
    assert raw["refuse"] is None, "guard=False measures the model alone"


def test_scene_guard_leaves_a_valid_request_alone():
    intent = scene_guard("set the table for one", _intent(place=["plate"]))
    assert intent["refuse"] is None
    assert intent["place"] == ["plate"]


def test_salvage_intent_from_truncated_json():
    truncated = '{"place": ["plate", "fork"], "pour": false, "open_drawer": null, "parap'
    assert not isinstance(extract_json(truncated), dict), "the outer object never closes"
    rescued = salvage_intent_fields(truncated)
    assert normalize_intent(rescued)["place"] == ["plate", "fork"]


def test_stage_a_salvage_runs_through_plan():
    stub = _Stub('Sure! {"place": ["plate"], "pour": true, "refuse": null, "paraph')
    result = plan("plate and a drink", _fake_world(), backend=stub)
    assert result["_meta"]["salvaged"] is True
    assert result["_meta"]["intent"]["place"] == ["plate"]
    assert result["_meta"]["intent"]["pour"] is True
    assert validate_plan(result, _fake_world()) == []


def test_stage_a_garbage_falls_back_to_keywords_and_says_so():
    stub = _Stub("I am a helpful assistant and I would love to help!")
    result = plan("set the table for one", _fake_world(), backend=stub)
    assert result["_meta"]["backend"] == "rules"
    assert result["_meta"]["fell_back_from"] == "stub"
    assert result["_meta"]["intent"]["place"] == list(PLACEABLE)
    assert validate_plan(result, _fake_world()) == []

    strict = plan("set the table for one", _fake_world(), backend=_Stub("nonsense"),
                  fallback_to_rules=False)
    assert strict["steps"] == [], "an unusable intent is a refusal, not two home steps"
    assert strict["refusal"]
    assert strict["_meta"]["fell_back_from"] is None


# --------------------------------------------------------------------------------------
# stage B — the deterministic expander
# --------------------------------------------------------------------------------------


def test_expand_full_setting_is_valid():
    world = _fake_world()
    result = expand(_intent(place=list(PLACEABLE)), world)
    assert validate_plan(result, world) == []
    skills = _skills(result)
    assert "open_drawer" in skills, "the cutlery is in the drawer"
    assert skills.count("handoff") == 4, "every prop starts out of the right arm's reach"
    assert skills.count("place") == 4
    assert skills[-2:] == ["home", "home"]
    placed = {s["args"]["target"] for s in result["steps"] if s["skill"] == "place"}
    assert placed == {"slot_plate", "slot_fork", "slot_spoon", "slot_mug"}


def test_expand_opens_the_drawer_only_for_drawer_objects():
    world = _fake_world()
    assert "open_drawer" not in _skills(expand(_intent(place=["plate"]), world))
    assert "open_drawer" in _skills(expand(_intent(place=["spoon"]), world))


def test_expand_skips_an_already_open_drawer():
    world = _fake_world()
    world["drawer"]["is_open"] = True
    result = expand(_intent(place=["fork"]), world)
    assert "open_drawer" not in _skills(result)
    assert "close_drawer" not in _skills(result), "we did not open it, we do not close it"


def test_expand_skips_objects_already_on_their_slot():
    world = _fake_world()
    world["objects"]["plate"]["on_slot"] = "slot_plate"
    result = expand(_intent(place=["plate", "mug"]), world)
    moved = {s["args"]["object"] for s in result["steps"] if s["skill"] == "pick"}
    assert moved == {"mug"}, "the plate is already done"


def test_expand_pour_sequence_order_and_staging():
    world = _fake_world()
    result = expand(_intent(pour=True), world)
    assert validate_plan(result, world) == []
    skills = _skills(result)
    assert skills.index("hold") < skills.index("pour")

    places = [s["args"] for s in result["steps"] if s["skill"] == "place"]
    assert {"arm": "right", "object": "mug", "target": "slot_mug"} in places
    assert {"arm": "left", "object": "bottle", "target": STAGING} in places, \
        "the bottle must not stay in the gripper"
    # The mug ends up on its slot, and nothing is held at the end.
    mug_place = next(i for i, s in enumerate(result["steps"])
                     if s["skill"] == "place" and s["args"]["object"] == "mug")
    assert mug_place > skills.index("pour")


def test_expand_does_not_lay_the_mug_out_twice_when_pouring():
    world = _fake_world()
    result = expand(_intent(place=list(PLACEABLE), pour=True), world)
    assert validate_plan(result, world) == []
    mug_picks = [s for s in result["steps"] if s["skill"] == "pick" and s["args"]["object"] == "mug"]
    assert len(mug_picks) == 1


def test_expand_leaves_an_explicitly_opened_drawer_open():
    world = _fake_world()
    result = expand(_intent(open_drawer=True), world)
    assert _skills(result) == ["open_drawer", "home", "home"]


def test_expand_honours_an_explicit_close():
    world = _fake_world()
    world["drawer"]["is_open"] = True
    assert "close_drawer" in _skills(expand(_intent(close_drawer=True), world))


def test_expand_frees_a_gripper_that_is_already_full():
    """Replanning mid-episode: an arm may be holding something we do not need."""
    world = _fake_world()
    world["arms"]["left"]["holding"] = "bottle"
    world["objects"]["bottle"]["held_by"] = "left"
    result = expand(_intent(place=["plate"]), world)
    assert validate_plan(result, world) == []
    assert any(s["skill"] == "place" and s["args"]["target"] == STAGING for s in result["steps"])


def test_expand_reuses_an_object_already_in_hand():
    world = _fake_world()
    world["arms"]["right"]["holding"] = "plate"
    world["objects"]["plate"]["held_by"] = "right"
    result = expand(_intent(place=["plate"]), world)
    assert validate_plan(result, world) == []
    assert "pick" not in _skills(result), "it is already held; do not pick it again"
    assert result["steps"][0]["skill"] == "place"


def test_expand_skips_a_pour_that_already_happened():
    """Replanning after the pour: fill it once, not twice."""
    world = _fake_world()
    world["water"] = {"in_mug": 4, "spilled": 0, "in_bottle": 6}
    world["arms"]["right"]["holding"] = "mug"
    world["objects"]["mug"]["held_by"] = "right"
    result = expand(_intent(pour=True), world)
    assert validate_plan(result, world) == []
    assert "pour" not in _skills(result)
    assert {"arm": "right", "object": "mug", "target": "slot_mug"} in \
        [s["args"] for s in result["steps"] if s["skill"] == "place"]


def test_expand_treats_an_occupied_slot_as_done():
    world = _fake_world()
    world["slots"]["slot_plate"]["occupied_by"] = "plate"
    result = expand(_intent(place=["plate"]), world)
    assert _skills(result) == ["home", "home"]


def test_expand_refusal_short_circuits():
    result = expand(_intent(refuse="there is no knife in the scene"), _fake_world())
    assert result["steps"] == []
    assert result["refusal"] == "there is no knife in the scene"


def test_every_step_has_a_rationale():
    result = expand(_intent(place=list(PLACEABLE), pour=True), _fake_world())
    assert all(s["rationale"].strip() for s in result["steps"])


def test_expander_self_check_raises_on_a_broken_plan(monkeypatch=None):
    """The self-check is the contract: a plan stage B cannot validate is our bug."""
    import tandem.planner.expander as expander_module

    original = expander_module._Expansion.finish
    try:
        # Emit a pick with a full gripper, which validate_plan must reject.
        def broken(self):
            self.add("pick", {"arm": "left", "object": "plate"}, "x")
            self.add("pick", {"arm": "left", "object": "mug"}, "x")

        expander_module._Expansion.finish = broken
        raised = False
        try:
            expand(_intent(place=["plate"]), _fake_world())
        except ExpansionError:
            raised = True
        assert raised, "expand must refuse to return a plan that does not validate"
    finally:
        expander_module._Expansion.finish = original


# --------------------------------------------------------------------------------------
# plan / replan
# --------------------------------------------------------------------------------------


def test_plan_end_to_end_reports_intent_and_stage_a_cost():
    world = _fake_world()
    result = plan("set the table for one and pour me some water", world, backend="rules")
    meta = result["_meta"]
    assert meta["valid"] is True, meta["errors"]
    assert meta["backend"] == "rules"
    assert meta["device"] == "CPU"
    assert "intent" in meta and meta["intent"]["pour"] is True
    assert meta["stage_a_ms"] >= 0.0
    assert validate_plan(result, world) == []


def test_plan_refusal_short_circuits_stage_b():
    world = _fake_world()
    result = plan("hand me the knife", world, backend="rules")
    assert result["steps"] == []
    assert "knife" in result["refusal"]
    assert result["_meta"]["refused"] is True
    assert result["_meta"]["valid"] is True


def test_replan_reuses_the_intent_and_skips_finished_work():
    world = _fake_world()
    first = plan("set the table for one", world, backend="rules")
    assert len(first["steps"]) == 16

    # The plate made it to its slot, then the fork was dropped mid-episode.
    world["objects"]["plate"]["on_slot"] = "slot_plate"
    world["drawer"]["is_open"] = True
    again = replan(first, world, executed_step_ids=[1, 2, 3, 4])

    assert again["_meta"]["replanned"] is True
    assert again["_meta"]["stage_a_ms"] == 0.0, "stage A must not run again"
    assert again["_meta"]["executed_step_ids"] == [1, 2, 3, 4]
    assert again["_meta"]["intent"] == first["_meta"]["intent"]
    assert validate_plan(again, world) == []

    picked = {s["args"]["object"] for s in again["steps"] if s["skill"] == "pick"}
    assert "plate" not in picked
    assert picked == {"fork", "spoon", "mug"}
    assert "open_drawer" not in _skills(again), "the drawer is already open"


def test_replan_when_everything_is_done():
    world = _fake_world()
    first = plan("put the plate out", world, backend="rules")
    world["objects"]["plate"]["on_slot"] = "slot_plate"
    again = replan(first, world, executed_step_ids=[1, 2, 3])
    assert [s["skill"] for s in again["steps"]] == ["home", "home"]


def test_replan_recovers_from_an_arm_holding_the_object():
    world = _fake_world()
    first = plan("put the plate out", world, backend="rules")
    world["arms"]["left"]["holding"] = "plate"
    world["objects"]["plate"]["held_by"] = "left"
    again = replan(first, world, executed_step_ids=[1])
    assert validate_plan(again, world) == []
    assert "pick" not in _skills(again)


def test_replan_carries_a_refusal():
    world = _fake_world()
    first = plan("get me a knife", world, backend="rules")
    again = replan(first, world, executed_step_ids=[])
    assert again["steps"] == []
    assert "knife" in again["refusal"]


# --------------------------------------------------------------------------------------
# validator
# --------------------------------------------------------------------------------------


def test_validate_catches_unknown_skill():
    errors = validate_plan(_numbered(_step("juggle", arm="left", object="mug")), _fake_world())
    assert any("unknown skill" in e for e in errors)


def test_validate_catches_bad_ids():
    world = _fake_world()
    duplicate = {"steps": [
        {"id": 1, "skill": "home", "args": {"arm": "left"}, "rationale": ""},
        {"id": 1, "skill": "home", "args": {"arm": "right"}, "rationale": ""},
    ]}
    errors = validate_plan(duplicate, world)
    assert any("1-based and contiguous" in e or "duplicate" in e for e in errors)

    missing = {"steps": [
        {"id": 1, "skill": "home", "args": {"arm": "left"}, "rationale": ""},
        {"id": 3, "skill": "home", "args": {"arm": "right"}, "rationale": ""},
    ]}
    assert any("id must be 2" in e for e in validate_plan(missing, world))


def test_validate_catches_pick_with_a_full_gripper():
    errors = validate_plan(
        _numbered(
            _step("pick", arm="left", object="mug"),
            _step("pick", arm="left", object="plate"),
        ),
        _fake_world(),
    )
    assert any("already holding" in e for e in errors)


def test_validate_catches_pour_with_no_holder():
    errors = validate_plan(
        _numbered(
            _step("pick", arm="left", object="bottle"),
            _step("pour", arm="left", source="bottle", target="mug"),
        ),
        _fake_world(),
    )
    assert any("nothing is holding" in e for e in errors)


def test_validate_catches_wrong_arg_keys_and_unknown_names():
    world = _fake_world()
    assert any("missing arg 'target'" in e
               for e in validate_plan(_numbered(_step("place", arm="left", object="mug")), world))
    assert any("unknown object 'knife'" in e
               for e in validate_plan(_numbered(_step("pick", arm="left", object="knife")), world))
    assert any("is not an arm" in e
               for e in validate_plan(_numbered(_step("pick", arm="middle", object="mug")), world))


def test_validate_accepts_staging_as_a_place_target():
    world = _fake_world()
    good = _numbered(
        _step("pick", arm="left", object="bottle"),
        _step("place", arm="left", object="bottle", target=STAGING),
    )
    assert validate_plan(good, world) == []
    bad = _numbered(
        _step("pick", arm="left", object="bottle"),
        _step("place", arm="left", object="bottle", target="slot_nowhere"),
    )
    assert any("unknown slot" in e for e in validate_plan(bad, world))


def test_validate_accepts_a_refusal_but_not_a_bare_empty_plan():
    assert validate_plan({"steps": [], "refusal": "no knife here"}, _fake_world()) == []
    assert validate_plan({"steps": []}, _fake_world()) == ["plan has no steps and no refusal"]


# --------------------------------------------------------------------------------------
# repair
# --------------------------------------------------------------------------------------


def test_repair_fixes_string_ids_and_the_cup_alias():
    sloppy = {
        "steps": [
            {"id": "1", "skill": "pick", "args": {"arm": "A", "object": "cup"}},
            {"id": "2", "skill": "handoff", "args": {"from": "A", "to": "B", "object": "cup"}},
            {"id": "3", "skill": "place", "args": {"arm": "B", "object": "cup", "target": "mug"}},
        ]
    }
    fixed = repair_plan(sloppy)
    assert [s["id"] for s in fixed["steps"]] == [1, 2, 3]
    assert fixed["steps"][0]["args"] == {"arm": "left", "object": "mug"}
    assert fixed["steps"][1]["args"] == {"from_arm": "left", "to_arm": "right", "object": "mug"}
    assert fixed["steps"][2]["args"]["target"] == "slot_mug"
    assert all(s["rationale"] == "" for s in fixed["steps"]), "missing rationale is filled in"
    assert validate_plan(fixed, _fake_world()) == []


def test_repair_maps_aside_to_staging():
    fixed = repair_plan({"steps": [
        {"id": 1, "skill": "pick", "args": {"arm": "left", "object": "bottle"}},
        {"id": 2, "skill": "place", "args": {"arm": "left", "object": "bottle", "target": "aside"}},
    ]})
    assert fixed["steps"][1]["args"]["target"] == STAGING
    assert validate_plan(fixed, _fake_world()) == []


def test_repair_renumbers_without_inventing_steps():
    fixed = repair_plan({"steps": [
        {"id": 7, "skill": "home", "args": {"arm": "left"}},
        {"id": 9, "skill": "home", "args": {"arm": "right"}},
    ]})
    assert [s["id"] for s in fixed["steps"]] == [1, 2]
    assert len(fixed["steps"]) == 2


def test_extract_json_survives_fences_and_chatter():
    raw = 'Sure! Here it is:\n```json\n{"place": [], "refuse": "nope"}\n```\nHope that helps.'
    assert extract_json(raw) == {"place": [], "refuse": "nope"}
    assert extract_json('{"pour": true} trailing junk') == {"pour": True}
    assert extract_json("no json at all") is None


# --------------------------------------------------------------------------------------
# prompt & selection
# --------------------------------------------------------------------------------------


def test_stage_a_prompt_is_small():
    system, user = build_intent_prompt("set the table for one")
    # Stage A has to be cheap; that is the whole point of the split.
    assert len(system) + len(user) < 1800, "stage-A prompt drifted over the ~450 token budget"
    assert "plate" in system and "refuse" in system
    assert user.count("instruction:") == 5, "four few-shot examples plus the real one"
    assert '"refuse": "there is no' in user, "one example must demonstrate a refusal"


def test_get_planner_honours_the_rules_choice():
    engine = get_planner("rules", cache=False)
    assert engine.info() == {
        "backend": "rules", "model": "keyword-parser", "device": "CPU", "precision": "n/a",
    }


def test_summarize_world_covers_every_entity():
    summary = summarize_world(_fake_world())
    for name in ("plate", "mug", "bottle", "spoon", "fork", "drawer", "slot_mug", "left arm"):
        assert name in summary


if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed, failed = 0, 0
    for name, func in tests:
        try:
            func()
        except Exception:  # noqa: BLE001 - this is the hand-rolled runner
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
