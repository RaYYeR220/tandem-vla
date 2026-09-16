# Stage-A intent parsing — scorecard

Fourteen instructions: ten that must parse to a specific intent, four that must be refused.
A case counts only if every pinned field matches. `[raw]` measures the model alone, with the
scene-inventory guard switched off.

| backend | intent | refusal | total | mean ms | median ms |
| --- | --- | --- | --- | --- | --- |
| rules (keyword parser) | 10/10 | 4/4 | 14/14 | 0.3 | 0.2 |
| planner-ov-0.5b INT4 | 5/10 | 4/4 | 9/14 | 1374.8 | 1375.4 |
| planner-ov-0.5b INT4 [raw] | 5/10 | 0/4 | 5/14 | 1282.4 | 1323.8 |
| planner-ov INT4 | 7/10 | 4/4 | 11/14 | 5513.6 | 4616.5 |
| planner-ov INT4 [raw] | 7/10 | 1/4 | 8/14 | 4203.8 | 3852.8 |

Two things this table is meant to settle, both of which shaped the design:

**A 1.5B model cannot plan this task, but it can parse it.** That is why planning is a
deterministic expansion over live world state and the model only answers one small question.
The 0.5B is four times faster and materially worse at it — it parrots the few-shot examples —
so the 1.5B stays the model default and the keyword parser stays the zero-dependency path.

**Neither model refuses reliably on its own** (1 of 4 and 0 of 4 in the `[raw]` rows). The scene
inventory is fixed and known, so refusing an absent object is a lookup rather than a judgement,
and it lives in code. That is what takes both models to 4 of 4 — and the `[raw]` rows are
published so the cost of that decision is visible rather than hidden.

> Latency was measured on a busy machine, with a video render running alongside. An earlier
> idle measurement of the same 1.5B export came in around 3.5 s. Accuracy is deterministic and
> identical across both runs; only the milliseconds move, and the slower figure is the one
> published here.

## Cases each backend missed

```
planner-ov-0.5b INT4 missed:
    'set the table for one': place=['mug', 'plate'] want ['fork', 'mug', 'plate', 'spoon']
    'set the table and pour me some water': place=['mug', 'plate'] want ['fork', 'mug', 'plate', 'spoon'] | ['fork', 'plate', 'spoon']
    'put the mug on the table': pour=True want False
    'I need a spoon and a fork': place=['fork', 'mug'] want ['fork', 'spoon']
    'put the plate down and fill the mug': refused: plate is already on the table

planner-ov-0.5b INT4 [raw] missed:
    'set the table for one': place=['mug', 'plate'] want ['fork', 'mug', 'plate', 'spoon']
    'set the table and pour me some water': place=['mug', 'plate'] want ['fork', 'mug', 'plate', 'spoon'] | ['fork', 'plate', 'spoon']
    'put the mug on the table': pour=True want False
    'I need a spoon and a fork': place=['fork', 'mug'] want ['fork', 'spoon']
    'put the plate down and fill the mug': refused: plate is already on the table
    'get me a knife': did not refuse
    'put a napkin next to the plate': did not refuse
    'pour me a glass of wine': did not refuse
    'bring me a bowl of soup': did not refuse

planner-ov INT4 missed:
    'put the mug on the table': place=[] want ['mug']
    'I need a spoon and a fork': place=['fork', 'plate', 'spoon'] want ['fork', 'spoon']
    'put the plate down and fill the mug': place=['fork', 'mug', 'plate', 'spoon'] want ['mug', 'plate'] | ['plate']

planner-ov INT4 [raw] missed:
    'put the mug on the table': place=[] want ['mug']
    'I need a spoon and a fork': place=['fork', 'plate', 'spoon'] want ['fork', 'spoon']
    'put the plate down and fill the mug': place=['fork', 'mug', 'plate', 'spoon'] want ['mug', 'plate'] | ['plate']
    'put a napkin next to the plate': did not refuse
    'pour me a glass of wine': did not refuse
    'bring me a bowl of soup': did not refuse
```
