"""Translate the SHOLA word list with NLLB-200 into every African language it covers.

    python3 nllb_translate.py --tier 1              # the commonest 11,206 words
    python3 nllb_translate.py --tier 1 --tier 2
    python3 nllb_translate.py --all                 # tiers 1-4, 71,014 words

Writes `out/nllb-tier<N>.jsonl`, one object per word per language, in the shape
`shola add-options` reads:

    {"phrase": "water", "language": "yor", "text": "omi"}

NLLB is encoder-decoder, so vLLM is no help here - this is plain transformers
with the model loaded once and `forced_bos_token_id` switched per target. The
expensive thing would be reloading 3.3B parameters 58 times; nothing else about
this is expensive.

Resumable at (tier, language-code) granularity: an interrupted run loses at most
the code in flight. Batches are sorted by token length so padding does not
dominate - these are noun phrases, and an unsorted batch wastes most of its
width on the longest member.

Licence note: the code here is ours, but NLLB-200's *weights* are CC-BY-NC-4.0.
Anything this produces inherits that, which is stricter than everything else in
the dataset.
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "facebook/nllb-200-3.3B"
SRC = "eng_Latn"
OUT = "out"
CHECKPOINT = "nllb-checkpoint.json"

# NLLB code -> the code SHOLA stores. Built by strict ISO-639-3 match against
# SHOLA's language list, with a short hand-checked fold so a specific variety
# lands on the code Google and Gemma already populated rather than opening a
# new, empty language nobody is being asked about:
#
#   aka -> twi  Akan macrolanguage; "twi" holds Asante, as Google's "ak" does
#   swh -> swa  Coastal Swahili
#   gaz -> orm  West Central Oromo
#   plt -> mlg  Plateau Malagasy
#   fuv -> ful  Nigerian Fulfulde
#   dik -> din  Southwestern Dinka
#
# Greek, Yiddish and Yemeni Arabic are in both lists and are dropped: SHOLA
# carries them because they are spoken somewhere in Africa, not because this
# project collects them. Spanish stays - it is official in Equatorial Guinea.
#
# Where a language has two scripts (Kanuri, Tamasheq) both are translated. They
# become two options on the same item, which is the point.
TARGETS = [
    ("aeb_Arab", "aeb"), ("afr_Latn", "afr"), ("amh_Ethi", "amh"),
    ("arb_Arab", "arb"), ("ary_Arab", "ary"), ("arz_Arab", "arz"),
    ("bam_Latn", "bam"), ("bem_Latn", "bem"), ("cjk_Latn", "cjk"),
    ("dik_Latn", "din"), ("dyu_Latn", "dyu"), ("ewe_Latn", "ewe"),
    ("fon_Latn", "fon"), ("fuv_Latn", "ful"), ("hau_Latn", "hau"),
    ("ibo_Latn", "ibo"), ("kab_Latn", "kab"), ("kam_Latn", "kam"),
    ("kbp_Latn", "kbp"), ("kea_Latn", "kea"), ("kik_Latn", "kik"),
    ("kin_Latn", "kin"), ("kmb_Latn", "kmb"), ("knc_Arab", "knc"),
    ("knc_Latn", "knc"), ("kon_Latn", "kon"), ("lin_Latn", "lin"),
    ("lua_Latn", "lua"), ("lug_Latn", "lug"), ("luo_Latn", "luo"),
    ("mos_Latn", "mos"), ("nso_Latn", "nso"), ("nus_Latn", "nus"),
    ("nya_Latn", "nya"), ("gaz_Latn", "orm"), ("plt_Latn", "mlg"),
    ("run_Latn", "run"), ("sag_Latn", "sag"), ("sna_Latn", "sna"),
    ("som_Latn", "som"), ("sot_Latn", "sot"), ("spa_Latn", "spa"),
    ("ssw_Latn", "ssw"), ("swh_Latn", "swa"), ("taq_Latn", "taq"),
    ("taq_Tfng", "taq"), ("tir_Ethi", "tir"), ("tsn_Latn", "tsn"),
    ("tso_Latn", "tso"), ("tum_Latn", "tum"), ("aka_Latn", "twi"),
    ("twi_Latn", "twi"), ("tzm_Tfng", "tzm"), ("umb_Latn", "umb"),
    ("wol_Latn", "wol"), ("xho_Latn", "xho"), ("yor_Latn", "yor"),
    ("zul_Latn", "zul"),
]


def load_words(path, tiers):
    want = set(tiers)
    out = {t: [] for t in tiers}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            phrase, _, tier = line.rstrip("\n").partition("\t")
            tier = int(tier)
            if tier in want:
                out[tier].append(phrase)
    return out


def read_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as fh:
            return set(tuple(x) for x in json.load(fh))
    return set()


def write_checkpoint(done):
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(sorted(done), fh)
    os.replace(tmp, CHECKPOINT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="words-kept.tsv")
    ap.add_argument("--tier", type=int, action="append", default=[])
    ap.add_argument("--all", action="store_true", help="tiers 1-4")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="first N words, for a pilot")
    args = ap.parse_args()

    tiers = sorted(set(args.tier)) or ([1, 2, 3, 4] if args.all else [1])
    words = load_words(args.words, tiers)
    for t in tiers:
        print(f"tier {t}: {len(words[t]):,} words", flush=True)
    if args.limit:
        for t in tiers:
            words[t] = words[t][:args.limit]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"loading {args.model} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, src_lang=SRC)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(OUT, exist_ok=True)
    done = read_checkpoint()
    total_pairs = len(tiers) * len(TARGETS)
    pair_n = 0

    for tier in tiers:
        phrases = words[tier]
        # Longest first: the first batch is the widest, so an out-of-memory
        # shows up in the first seconds rather than an hour in.
        order = sorted(range(len(phrases)), key=lambda i: -len(phrases[i]))
        path = os.path.join(OUT, f"nllb-tier{tier}.jsonl")
        for code, language in TARGETS:
            pair_n += 1
            key = (tier, code)
            if key in done:
                print(f"[{pair_n}/{total_pairs}] tier {tier} {code} - done",
                      flush=True)
                continue
            bos = tok.convert_tokens_to_ids(code)
            started = time.time()
            n = 0
            with open(path, "a", encoding="utf-8") as out:
                for start in range(0, len(order), args.batch_size):
                    chunk = [phrases[i] for i in
                             order[start:start + args.batch_size]]
                    enc = tok(chunk, return_tensors="pt", padding=True,
                              truncation=True, max_length=128).to("cuda:0")
                    with torch.inference_mode():
                        gen = model.generate(
                            **enc, forced_bos_token_id=bos,
                            max_new_tokens=args.max_new_tokens, num_beams=1)
                    texts = tok.batch_decode(gen, skip_special_tokens=True)
                    for phrase, text in zip(chunk, texts):
                        text = (text or "").strip()
                        if text and text.lower() != phrase.lower():
                            out.write(json.dumps(
                                {"phrase": phrase, "language": language,
                                 "text": text}, ensure_ascii=False) + "\n")
                            n += 1
                    if start and start % (args.batch_size * 20) == 0:
                        rate = (start + len(chunk)) / (time.time() - started)
                        print(f"    {code} {start + len(chunk):,}/"
                              f"{len(order):,}  {rate:.0f}/s", flush=True)
            took = time.time() - started
            print(f"[{pair_n}/{total_pairs}] tier {tier} {code} -> {language}  "
                  f"{n:,} lines  {took / 60:.1f}m  "
                  f"{len(order) / took:.0f}/s", flush=True)
            done.add(key)
            write_checkpoint(done)

    print("all done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
