# What the SHOLA run produced

The sentences were loaded into SHOLA on 7 September 2026 and taken back down on
23 September when the platform narrowed to words only. This is everything the
volunteers produced in that window, kept because it is human work and cannot be
regenerated.

## Which machine translation speakers agreed with

| System | Offered | Picked | Rate | Its own |
| --- | ---: | ---: | ---: | ---: |
| `gemini-3.6-flash` | 19 | 12 | **63%** | 10 |
| `google-translate` | 19 | 8 | 42% | 3 |
| `google-translate-via-thai` | 20 | 5 | 25% | 2 |

Head to head, counting only the items where they proposed different English:

| | |
| --- | --- |
| gemini-3.6-flash **12–4** google-translate-via-thai | of 19 compared |
| gemini-3.6-flash **10–6** google-translate | of 17 compared |
| google-translate **5–1** google-translate-via-thai | of 12 compared |

Twenty answers is far too few to call this settled - one speaker, one language,
a single sitting. Read it as the shape of a result rather than a result: Gemini
ahead, the direct Google pass behind it, and the Thai pass-through clearly
worst, which is what the pivot's job was always likely to cost it.

The one thing it does establish is that the three legs disagree often enough to
be worth asking about. On 19 of the items Gemini and the Thai pivot proposed
different English, and a speaker chose between them.

## Files

- `ghana-sentences-answers.csv` - every answer with its vote count
- `ghana-sentences-model-scores.csv` - the table above, as published by the API
- `ghana-sentences-shola.csv` - the 36,414 items as they were loaded
- `gemini-english.jsonl.gz`, `google-english.jsonl.gz` - the raw translations
