# Google Translate seed, tier 1

604,160 translations of the 11,206 commonest English nouns into the **55
African languages Google Translate supports**, for `shola add-options`.

    flask --app wsgi shola add-options \
      --jsonl google-tier1.jsonl.gz --source google-translate

## Why

2,202 of SHOLA's 2,206 languages arrive with nothing in them, so the first
speaker to answer types every wording into a blank box. A machine translation
to agree with or correct is a far smaller ask, and it gives the model
scoreboard something to measure beyond the four languages seeded by hand.

## What is in it

One line per word per language, `{"phrase","language","text"}`, keyed to
SHOLA's own language codes. Zero malformed rows, zero failed batches.

A translation identical to the English is dropped rather than written - 12,085
of them - because as an option it tells a speaker nothing. That is why the
per-language counts sit below 11,206, Shona lowest at 8,775.

## Quality

Spot-checked on common words, most languages look sound:

| | water | house | mother |
| --- | --- | --- | --- |
| Swahili | maji | nyumba | mama |
| Hausa | ruwa | gida | uwa |
| Zulu | amanzi | indlu | umama |
| Yoruba | omi | ile | iya |
| Tamazight | ⴰⵎⴰⵏ | ⴰⵅⵅⴰⵎ | ⵜⴰⵢⴻⵎⴰⵜ |
| Twi | nsuo | fie | maame |

Fon is the one to watch: `xwé` and `nɔ` are right, but "water" came back as
`nukɔn nukɔntɔn ɔ` when it should be `sìn`. Wrong options are not fatal here -
a speaker corrects one and their wording becomes what the next speaker votes
on - but they do put visibly wrong text in front of people, and a language
whose seed is mostly wrong may be better left empty.

`ber` is Google's collective Berber code and maps to `zgh`, Standard Moroccan
Tamazight: every character it returns is Tifinagh, which is what `zgh` is
written in. See the note in `scripts/google-seed/translate_words.py`.
