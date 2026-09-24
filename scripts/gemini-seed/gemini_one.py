"""Re-translate the SHOLA word list with Gemini, one wording per item.

    GEMINI_KEY=... python3 gemini_one.py --all

Gemini was originally asked for three candidate translations per word, while
Google, Gemma and NLLB each give one. That is not a like-for-like comparison:
three wordings get three chances to match what a speaker says, and Gemini's
86% pick rate against Google's 34% partly measures the extra chances rather
than better translation. This asks for the single best wording instead, so the
scoreboard compares one answer against one answer.

Writes `out/gemini-one.jsonl` in the shape `shola add-options` reads:

    {"phrase": "water", "language": "twi", "text": "nsuo"}

Resumable: keyed on "<language>:<phrase>", already-written lines are skipped.
The key is read from GEMINI_KEY and never written to disk or logged.
"""

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["GEMINI_KEY"]
MODEL = "gemini-3.6-flash"
URL = (f"https://generativelanguage.googleapis.com/v1beta/models/"
       f"{MODEL}:generateContent?key={KEY}")
OUT = "out/gemini-one.jsonl"

# The four the original run covered, and so the four whose options need
# replacing. SHOLA's "twi" holds Asante.
LANG = {"twi": "Asante Twi", "ewe": "Ewe", "gaa": "Ga", "dag": "Dagbani"}

BATCH = 40
lock = threading.Lock()
written = 0


def load_done():
    seen = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    seen.add((d["language"], d["phrase"]))
                except Exception:
                    pass
    return seen


def prompt_for(language, words):
    numbered = "\n".join(f"{i + 1}. {w}" for i, w in enumerate(words))
    return (
        f"You are a professional English to {language} translator. "
        f"{language} is a Ghanaian language.\n\n"
        f"Translate each numbered English term below into {language}.\n\n"
        f"Rules:\n"
        f"- Give exactly ONE translation per term: the single wording a "
        f"{language} speaker would most naturally use. Do not offer "
        f"alternatives, do not use slashes or brackets, do not explain.\n"
        f"- Translate meaning, not word for word.\n"
        f"- These are noun phrases from news, research and speech. Keep the "
        f"result a noun phrase.\n"
        f"- If a term is a proper name or has no {language} equivalent, return "
        f"it unchanged.\n"
        f"- Return ONLY a JSON array of {len(words)} strings, in the same "
        f"order, with no other text.\n\n"
        f"{numbered}"
    )


def call(text, tries=6):
    body = json.dumps({
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
    }).encode()
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
            return d["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:                       # noqa: BLE001
            last = e
            code = getattr(e, "code", None)
            if code in (400, 403, 404):
                raise
            time.sleep(min(60, 2 ** attempt) + random.random() * 2)
    raise RuntimeError(f"gave up: {last}")


def parse(text, n):
    """The JSON array Gemini was asked for, however it wrapped it."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        got = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(got, list) or len(got) != n:
        return None
    return [str(x).strip() for x in got]


def run_batch(code, words, fh):
    global written
    language = LANG[code]
    try:
        got = parse(call(prompt_for(language, words)), len(words))
    except Exception as e:                           # noqa: BLE001
        print(f"  {code} batch failed: {e}", flush=True)
        return
    if got is None:
        # One bad batch should not cost 40 words. Retry them singly.
        got = []
        for w in words:
            try:
                one = parse(call(prompt_for(language, [w])), 1)
                got.append(one[0] if one else "")
            except Exception:
                got.append("")
    rows = []
    for w, t in zip(words, got):
        t = (t or "").strip()
        if t and t.lower() != w.lower():
            rows.append(json.dumps({"phrase": w, "language": code, "text": t},
                                   ensure_ascii=False))
    if rows:
        with lock:
            fh.write("\n".join(rows) + "\n")
            fh.flush()
            written += len(rows)
            if written % 4000 < len(rows):
                print(f"  {written:,} written", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", default="words-kept.tsv")
    ap.add_argument("--tier", type=int, action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tiers = set(args.tier) or ({1, 2, 3, 4} if args.all else {1})
    phrases = []
    with open(args.words, encoding="utf-8") as fh:
        for line in fh:
            phrase, _, tier = line.rstrip("\n").partition("\t")
            if int(tier) in tiers:
                phrases.append(phrase)

    done = load_done()
    todo = [(c, p) for c in LANG for p in phrases if (c, p) not in done]
    print(f"{len(phrases):,} words x {len(LANG)} languages = "
          f"{len(phrases) * len(LANG):,} translations")
    print(f"already done: {len(done):,}")
    print(f"to do: {len(todo):,} in {-(-len(todo) // BATCH):,} requests",
          flush=True)
    if not todo:
        return

    os.makedirs("out", exist_ok=True)
    jobs = []
    by_code = {}
    for code, phrase in todo:
        by_code.setdefault(code, []).append(phrase)
    for code, words in by_code.items():
        for i in range(0, len(words), BATCH):
            jobs.append((code, words[i:i + BATCH]))
    random.shuffle(jobs)

    with open(OUT, "a", encoding="utf-8") as fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda j: run_batch(j[0], j[1], fh), jobs))
    print(f"done, {written:,} lines", flush=True)


if __name__ == "__main__":
    sys.exit(main())
