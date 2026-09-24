# NLLB-200 seed

Translates the SHOLA word list into the 55 African languages NLLB-200 covers,
adding a third machine opinion beside Google Translate and Gemma 4 31B.

```
python3 nllb_translate.py --all --batch-size 256
```

Writes `out/nllb-tier<N>.jsonl` in the shape `shola add-options` reads.

## Why transformers and not vLLM

NLLB is encoder-decoder. vLLM is no help, so this is plain `transformers` with
the 3.3B model loaded once and `forced_bos_token_id` switched per target — the
only expensive thing would be reloading 3.3B parameters 58 times.

Measured on the H200 at batch 256, bf16: **~1,300 words/s per language**, so
71,014 words across 58 codes is about an hour.

## Coverage

NLLB-200 has 202 script-codes over 196 distinct languages. Only **55** are
languages SHOLA collects — the rest are German, Japanese, Hindi and so on. Of
those 55, **33 were already covered by Google Translate**, so NLLB's real
contribution is split two ways: a second opinion on the 33, and first coverage
of ~15 that nothing else reached (Chokwe, Kabyle, Kamba, Kabiyè,
Kabuverdianu, Kikuyu, Kimbundu, Mossi, Umbundu, Tamasheq, Central Atlas
Tamazight, and three Arabic varieties).

It also *loses* 22 that Google has — Ga, Krio, Afar, Acholi, Fula, Malagasy,
Tiv, Venda, Baoulé and the two French creoles among them. Neither system is a
superset of the other.

Where a language has two scripts (Kanuri, Tamasheq) both are translated; they
become two options on the same item, which is the point. Six NLLB codes are
folded onto the code Google and Gemma already populated (`aka`→`twi`,
`swh`→`swa`, `gaz`→`orm`, `plt`→`mlg`, `fuv`→`ful`, `dik`→`din`) so the output
lands beside theirs rather than opening a new, empty language.

## Licence

The code here is ours. NLLB-200's **weights are CC-BY-NC-4.0**, and anything
this produces inherits that — stricter than everything else in the dataset.
