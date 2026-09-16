"""Builds ``results/eval/scorecard.json`` on demand, for ``GET /api/scorecard``.

Reuses ``tandem.eval.run_eval`` verbatim (that module is owned by the eval track and is not
touched here). It is called with ``out_dir=None`` deliberately: the per-episode dump
(``episodes.json``) hits a pre-existing serialization gap in that code path — a stray
``numpy.bool_`` surviving a ``bool(...) and <numpy comparison>`` expression a few layers down
in the control stack — which is out of scope to patch from here. The aggregate ``summary``
dict returned by ``run_eval`` does not go through that path (it is built from plain
``sum()``/``round()`` over already-native fields), so it serializes cleanly, and this module
writes it out itself with the same numpy-tolerant encoder the rest of the server uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..eval.run_eval import EvalConfig, run_eval, to_markdown
from .jsonutil import dumps

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "eval"


def build_scorecard(
    *,
    seeds: int = 5,
    seed0: int = 0,
    dr_scale: float = 1.0,
    budget_s: float = 150.0,
    no_pour: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run ``seeds`` episodes of the canonical task and write the scorecard, failures included."""
    out_dir = out_dir or DEFAULT_OUT_DIR
    intent = {"place": ["plate", "fork", "spoon", "mug"], "pour": not no_pour}
    cfg = EvalConfig(
        seeds=tuple(range(seed0, seed0 + seeds)),
        instruction=(
            "set the table for one" if no_pour else "set the table for one and pour me some water"
        ),
        intent=intent,
        dr_scale=dr_scale,
        budget_s=budget_s,
    )
    summary = run_eval(cfg, out_dir=None, verbose=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(dumps(summary), encoding="utf-8")
    (out_dir / "scorecard.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def read_scorecard(out_dir: Path | None = None) -> dict[str, Any] | None:
    path = (out_dir or DEFAULT_OUT_DIR) / "scorecard.json"
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))
