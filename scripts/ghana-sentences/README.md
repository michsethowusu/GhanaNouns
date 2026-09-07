# Translating `ghanaopenai/ghana-sentences` into English

Three candidate English translations per sentence, so that the choice between
them is something Ghanaian speakers can settle on SHOLA rather than something
we decide here.

| Leg | Languages | What it is |
| --- | --- | --- |
| `gemini_leg.py` | all 12 subsets | `gemini-3.6-flash`, batched 12 sentences per call |
| `google_leg.py` | Twi, Ewe, Ga | Google Translate free endpoint, source → English |
| `google_leg.py` | Twi, Ewe, Ga | the same endpoint, source → Thai → English |

## The two Twi varieties

The dataset keeps `twi-aku` and `twi-asa` apart, and so does the SHOLA project:
they land on `twi-akuapem` and `twi` respectively. The same English sentence can
then collect an Akuapem answer and an Asante answer, which is the useful
outcome; merging them would have thrown one of the two away.

The translation legs treat them differently, because the tools do. Gemini is
told which variety it is reading. Google Translate has only `ak`, so both are
sent as `ak` and the two varieties get the same Google candidates — the
distinction there is in who evaluates them, not in what was proposed.

The Thai pass-through is there because it fails differently. Where the direct
translation quietly guesses, the pivot usually guesses something else, and two
disagreeing candidates are more useful to a speaker than one confident wrong
answer.

## Running

    export SCR=work GEMINI_KEY=...        # never committed
    python3 gemini_leg.py                 # ~35 min at 8 workers
    python3 google_leg.py                 # ~110 min at 4 workers
    python3 build_csv.py                  # writes the SHOLA project CSV

Both legs append to `$SCR/out/*.jsonl` keyed by `<subset>:<row>` and skip what
is already there, so either can be stopped and re-run: a run that dies halfway
costs the batches in flight, not the work already done.

## Notes on the endpoints

`translate_a/single` returns Google's abuse page regardless of user agent.
`clients5.google.com/translate_a/t?client=dict-chrome-ex` works, honours `sl`
(checked against a control: `sl=de` returns Twi text untranslated), and accepts
repeated `q` parameters, which is what makes ten sentences per request possible
instead of one.

## What gets excluded

The dataset was extracted from curriculum PDFs and roughly half of what came
out is not a sentence anybody can translate. Those rows are excluded rather
than translated: a speaker shown half a sentence can neither answer it nor
usefully skip it.

| Excluded | Rows | Why |
| --- | --- | --- |
| starts mid-sentence | 21,711 | a line beginning in lower case is the tail of a sentence wrapped onto the previous line |
| cut off | 13,851 | no closing punctuation, *and* the next line starts mid-sentence — so the continuation is right there in the file |
| junk or a reference | 641 | URLs, query strings, dotted table-of-contents leaders, page numbers |
| too short | 314 | under 15 characters |
| duplicate text | 274 | see below |

That leaves **~32,900 of 73,556**, and it keeps the 4,276 `nsanku-mmlu` rows,
which are well-formed sentences throughout.

The "cut off" test is worth spelling out. A truncated line and a heading look
identical on their own — both short, both unpunctuated. What separates them is
the *next* line: if it starts mid-sentence, the line before it was cut off.
That evidence is in the file, so it is used instead of guessing from length,
and genuine headings survive.

## Other decisions

- **One item per text.** 337 sentences appear more than once and 131 of those
  span two subsets. In SHOLA an item exists once, so a duplicate kept in both
  places would be filed under no single language and shown to every speaker.
- **The item is filed under its own language.** A Twi sentence with English
  options is a Twi item; Ewe speakers are never asked to read it.
- **A translation identical to its input is discarded** — that is the endpoint
  giving up, not an answer.
- **Identical wordings collapse into one option** carrying both system names,
  separated by `;`. Showing a speaker the same option twice is a worse question,
  and SHOLA credits both systems when that option is picked.
