"""Translate the SHOLA word list into every African language Google supports.

    python3 translate_words.py --tier 1            # the commonest 11,206 words
    python3 translate_words.py --tier 1 --tier 2

Writes `out/google-<tier>.jsonl`, one object per word per language, in the
shape `shola add-options` reads:

    {"phrase": "water", "language": "yor", "text": "omi"}

Most African languages arrive in SHOLA with nothing in them, so the first
speaker types every wording from scratch. A machine translation to agree with
or correct is a much smaller ask than a blank box, and it gives the scoreboard
something to measure across more than the four languages seeded by hand.

Resumable: every line already written is skipped, so an interrupted run costs
the batches in flight and nothing else. Google's free endpoint is the same one
the sentences run used - it takes repeated `q` parameters, which is what makes
twenty words per request possible instead of one.
"""

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ENDPOINT = "https://clients5.google.com/translate_a/t"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
URL_BUDGET = 1500
WORDS_PER_REQUEST = 20

# Google's code -> the code SHOLA stores. Checked against afriso: every one of
# these resolves to a language SHOLA lists, and every one was confirmed to
# actually translate rather than hand the English back.
#
# `ber` is the odd one. It is ISO 639-2/5's *collective* code for the whole
# Berber family, so it names no single language and cannot map to itself.
# Google labels it "Tamazight (Tifinagh)" and every character it returns is
# Tifinagh - 46 of 46 letters across a test of eight words. That is Standard
# Moroccan Tamazight, `zgh`: the standardised variety that has been official in
# Morocco since 2011 and is written in Tifinagh by design.
#
# The alternatives fit worse. Kabyle (`kab`) is written overwhelmingly in Latin
# script, which the output rules out. Tachelhit (`shi`) and Central Atlas
# Tamazight (`tzm`) are specific regional varieties, and filing standardised
# output under one of them would tell a speaker it is something it is not.
LANGUAGES = {
    "ach": "ach", "aa": "aar", "af": "afr", "alz": "alz", "am": "amh",
    "ak": "twi", "bm": "bam", "bci": "bci", "bem": "bem", "ny": "nya",
    "cgg": "cgg", "din": "din", "dov": "dov", "dyu": "dyu", "ee": "ewe",
    "fon": "fon", "ff": "ful", "gaa": "ga", "lg": "lug", "ha": "hau",
    "ig": "ibo", "kr": "kau", "rw": "kin", "ktu": "ktu", "kg": "kon",
    "kri": "kri", "ln": "lin", "lua": "lua", "luo": "luo", "mg": "mlg",
    "mfe": "mfe", "nus": "nus", "om": "orm", "nso": "nso", "rn": "run",
    "sg": "sag", "crs": "crs", "sn": "sna", "so": "som", "nr": "nbl",
    "st": "sot", "sus": "sus", "sw": "swa", "ss": "ssw", "ti": "tir",
    "tiv": "tiv", "ts": "tso", "tn": "tsn", "tum": "tum", "ve": "ven",
    "wo": "wol", "xh": "xho", "yo": "yor", "zu": "zul", "ber": "zgh",
}

lock = threading.Lock()
done_n = 0


def tiers_of(freq_csv):
    """(phrase, tier) for every word, tier 1 being the commonest.

    The same thresholds tiers.py uses, so "tier 1" means the same thing here as
    it does on the progress page.
    """
    out = []
    with open(freq_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n = sum(float(row[k] or 0) for k in
                    ("news_count", "research_count", "speech_count"))
            tier = 1 if n >= 50 else 2 if n >= 20 else 3 if n >= 10 \
                else 4 if n >= 5 else 5
            out.append((row["phrase"], tier))
    return out


def batches(words):
    """Group words so the encoded URL stays inside URL_BUDGET."""
    cur, size = [], 0
    for w in words:
        cost = len(urllib.parse.quote(w)) + 3
        if cur and (size + cost > URL_BUDGET or len(cur) >= WORDS_PER_REQUEST):
            yield cur
            cur, size = [], 0
        cur.append(w)
        size += cost
    if cur:
        yield cur


def translate(texts, target, tries=6):
    """One English word in, one translation out, order preserved."""
    params = [("client", "dict-chrome-ex"), ("sl", "en"), ("tl", target)]
    params += [("q", t) for t in texts]
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            got = []

            def flatten(node):
                if isinstance(node, str):
                    got.append(node)
                elif isinstance(node, list):
                    for x in node:
                        flatten(x)

            flatten(data)
            if len(got) == len(texts):
                return got
            last = f"got {len(got)} for {len(texts)}"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except Exception as e:                      # noqa: BLE001
            last = repr(e)
        time.sleep(min(90, 3 * 2 ** attempt) + random.random() * 4)
    if len(texts) == 1:
        raise RuntimeError(f"gave up: {last}")
    mid = len(texts) // 2                           # bisect on a mismatch
    return (translate(texts[:mid], target, tries)
            + translate(texts[mid:], target, tries))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq-csv", default="data/ghana-nouns.csv")
    ap.add_argument("--tier", type=int, action="append", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", help="comma-separated Google codes, for testing")
    args = ap.parse_args()
    wanted_tiers = set(args.tier or [1])

    words = [p for p, t in tiers_of(args.freq_csv) if t in wanted_tiers]
    codes = {g: s for g, s in LANGUAGES.items()
             if not args.only or g in args.only.split(",")}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out,
                        "google-" + "-".join(map(str, sorted(wanted_tiers)))
                        + ".jsonl")

    seen = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    seen.add((row["phrase"], row["language"]))
                except Exception:                   # noqa: BLE001
                    pass
    print(f"{len(words):,} words x {len(codes)} languages "
          f"= {len(words) * len(codes):,} translations")
    print(f"already done: {len(seen):,}")

    jobs = []
    for gcode, scode in sorted(codes.items()):
        pending = [w for w in words if (w, scode) not in seen]
        jobs += [(gcode, scode, b) for b in batches(pending)]
    random.shuffle(jobs)
    print(f"requests to make: {len(jobs):,}", flush=True)
    started = time.time()

    def run(job):
        global done_n
        gcode, scode, chunk = job
        try:
            got = translate(chunk, gcode)
        except Exception as exc:                    # noqa: BLE001
            sys.stderr.write(f"{gcode} batch failed: {exc}\n")
            return
        lines = []
        for word, text in zip(chunk, got):
            text = (text or "").strip()
            # A "translation" identical to the English is the endpoint giving
            # up. Some are legitimately the same word, but as an option it
            # tells a speaker nothing, so it is not written.
            if not text or text.casefold() == word.casefold():
                continue
            lines.append(json.dumps({"phrase": word, "language": scode,
                                     "text": text}, ensure_ascii=False))
        with lock:
            if lines:
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
            done_n += len(chunk)
            if done_n % 20000 < len(chunk):
                rate = done_n / max(1, time.time() - started)
                print(f"  {done_n:,} translated  ({rate:,.0f}/s)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(run, jobs))
    print(f"done: {done_n:,} translated into {path}")


if __name__ == "__main__":
    main()
