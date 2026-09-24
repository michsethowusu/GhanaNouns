# Gemini single-wording rerun

```
GEMINI_KEY=... python3 gemini_one.py --all --workers 8
```

Writes `out/gemini-one.jsonl` in the shape `shola add-options` reads. The key
comes from the environment and is never written to disk or logged.

## Why

The first Gemini run asked for **three** candidate translations per word.
Google Translate, Gemma 4 31B and NLLB-200 each give **one**:

| System | Options per item |
|--------|------------------|
| gemini-3.6-flash | 3 |
| google-translate | 1 |
| gemma-4-31b-it | 1 |
| nllb-200-3.3B | 1 |

Three wordings get three chances to match what a speaker picks. Gemini's 86%
pick rate against Google's 34% was therefore measuring the extra chances as
much as the quality of the translation — the scoreboard was not comparing like
with like.

This asks for the single wording a speaker would most naturally use, so every
system is judged on one answer.

## Replacing the old options

Two steps, each of which counts before it acts:

```
flask shola retract-options --source gemini-3.6-flash          # count
flask shola retract-options --source gemini-3.6-flash --yes
flask shola add-options --jsonl gemini-one.jsonl --source gemini-3.6-flash --yes
```

`retract-options` takes the name off the old options. It does **not** delete an
option a speaker has already answered against — a `Candidate` a verdict points
at holds the wording the other systems are being compared against, so deleting
it would take their scores down too. Those rows stay, marked `retracted`, which
the scoreboard counts for nobody.

A wording that comes back identical is credited again and keeps its score. One
that comes back different does not.

## Scope

71,014 words × 4 languages (Asante Twi, Ewe, Ga, Dagbani) = 284,056
translations, about 7,100 requests at 40 words each.
