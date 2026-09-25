# Gemma 4 31B — extra African languages

Extends the Gemma run from 55 languages to the ones it can actually handle out
of a 282-language African list (from JW.org's language inventory).

```
python3 gemma_pilot2.py        # echo test
python3 gemma_consist.py       # self-consistency test
python3 consist_verdict2.py    # writes jw-final-langs.tsv
python3 gemma_jw.py            # the run
python3 upload_jw.py --watch   # publish each tier as it finishes
```

## Why there is a filter at all

A large language model produces confident, fluent output for a language it has
never been trained on. Asked for Abua it will return *something*, and that
something looks like a translation. Handing it to a speaker wastes the one
resource this project cannot buy more of.

So the list was filtered before anything was translated.

**Echo test** — translate 40 everyday words and count how often the English
comes back unchanged. A model with nothing to say echoes. This caught 25
languages, including Nigerian Pidgin, Liberian English and Cameroonian Pidgin,
where echoing is arguably correct but tells a speaker nothing.

It was not enough. 257 of 282 passed, including languages with a few thousand
speakers and no written corpus. Echo only catches a model that gives up, not
one that invents.

**Self-consistency** — sample the same 20 words three times at temperature 0.8
and compare. A word the model knows comes back the same each time; one it is
inventing wanders. Six languages Gemma demonstrably handles (Swahili, Zulu,
Hausa, Shona, Lingala, Yoruba) scored 15–20 out of 20 and set the bar at 50%.

This removed a further 182, leaving **75**.

## Compare on normalised forms

The first version of the consistency test compared raw strings and scored
Yoruba **2/20** — while it was returning `omi` for water every single time. Its
tone marks moved between samples. Case-folding and stripping combining marks
put Yoruba at 15/20 and made the measure mean what it claims to.

Had that gone unnoticed, the bar would have been set from a broken control at
1/20 and almost every language would have passed.

## Even so

A language can pass both tests and still be partly invented. Nothing here is a
verified translation; it is a prompt for a speaker to correct.

`jw-final-langs.tsv` is the list that survived.
