"""Turn the three translation legs into one SHOLA project CSV.

Columns: text,language,priority,option1..3,model1..3

`text` is the Ghanaian sentence and the options are candidate English
translations, so the item is filed under the language it is written in - Twi
speakers see Twi sentences, and nobody is asked to read a language they may not
have. Where two systems produced the same English, the option is written once
with both names against it rather than twice on screen.
"""
import csv, json, os, re, sys
import pandas as pd

SCR = os.environ.get("SCR", "work")
GEMINI = "gemini-3.6-flash"
DIRECT = "google-translate"
PIVOT = "google-translate-via-thai"

# Which SHOLA language code each subset's speakers are.
CODE = {"ada": "ada", "dag": "dagbani", "dga": "dga", "ewe": "ewe",
        "fat": "fat", "gaa": "ga", "gjn": "gjn", "gur": "gur", "nzi": "nzi",
        "twi-aku": "twi-akuapem", "twi-asa": "twi", "xsm": "xsm"}

# The dataset was extracted from curriculum PDFs, and roughly half of what came
# out is not a sentence anybody can translate. Those rows are excluded rather
# than translated, because volunteer attention is the one thing this project
# cannot buy more of, and a speaker asked to translate half a sentence can
# neither answer nor usefully skip it.
JUNK = re.compile(
    r"https?://|www\.|\.com/|&pid=|&ie=|fpstate=|&sres|vld=cid:"
    r"|\.\.+|[\u2022|]|\d{3,}", re.I)
STARTS_LOWER = re.compile(r"^[a-z\u025b\u0254\u014b\u028b\u0256\u0192\u0263]")
ENDS_SENTENCE = re.compile(r"[.!?\u2026\u00bb\"'\u201d]$")
LETTER = re.compile(r"[A-Za-z\u025b\u0254\u014b\u028b\u0256\u0192\u0263\u0190\u0186\u014a]")
MIN_LEN = 15


def exclusions(df):
    """Why each row is unusable, or None. Keyed by "<subset>:<row>".

    The tests are cheap and each one names a way the extraction failed:

    - **starts mid-sentence** - a line beginning in lower case is the tail of a
      sentence that was wrapped onto the previous line.
    - **cut off** - a line with no closing punctuation whose *next* line starts
      mid-sentence was itself cut off. That the continuation exists in the file
      is much better evidence than the line's length: it distinguishes a
      truncated line from a heading, which is short and unpunctuated too but
      perfectly translatable.
    - **junk or a reference** - URLs, query strings, dotted table-of-contents
      leaders, page numbers.
    """
    why = {}
    for config, sub in df.groupby("config", sort=True):
        texts = sub.text.tolist()
        low = [bool(STARTS_LOWER.match(t)) for t in texts]
        for i, text in enumerate(texts):
            uid = f"{config}:{i}"
            nxt = low[i + 1] if i + 1 < len(low) else False
            if len(text) < MIN_LEN:
                why[uid] = "too short"
            elif JUNK.search(text):
                why[uid] = "junk or a reference"
            elif not LETTER.search(text):
                why[uid] = "no letters"
            elif low[i]:
                why[uid] = "starts mid-sentence"
            elif not ENDS_SENTENCE.search(text) and nxt:
                why[uid] = "cut off, the next line continues it"
    return why


def load(path, keys):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[row["uid"]] = {k: (row.get(k) or "").strip() for k in keys}
    return out


def same(a, b):
    return a.casefold().strip(" .!?") == b.casefold().strip(" .!?")


def main():
    df = pd.read_parquet(f"{SCR}/gs/all.parquet")
    gem = load(f"{SCR}/out/gemini.jsonl", ["en"])
    goo = load(f"{SCR}/out/google.jsonl", ["direct", "pivot"])

    dropped = exclusions(df)

    rows = []
    seen = set()
    stats = {"duplicate": 0, "no_translation": 0, "written": 0}
    excluded = {}
    per_lang = {}
    for config, sub in df.groupby("config", sort=True):
        code = CODE[config]
        for i, text in enumerate(sub.text.tolist()):
            uid = f"{config}:{i}"
            reason = dropped.get(uid)
            if reason:
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            key = text.casefold()
            if key in seen:            # one item per text, so it stays in one
                stats["duplicate"] += 1   # language and one speaker group
                continue

            # Collapse identical wordings, keeping every system that produced
            # them, in the order the systems are worth showing.
            candidates = []
            for name, value in ((GEMINI, gem.get(uid, {}).get("en", "")),
                                (DIRECT, goo.get(uid, {}).get("direct", "")),
                                (PIVOT, goo.get(uid, {}).get("pivot", ""))):
                value = value.strip()
                # A "translation" identical to the input is the endpoint giving
                # up, not an answer.
                if not value or same(value, text):
                    continue
                for existing in candidates:
                    if same(existing["text"], value):
                        existing["models"].append(name)
                        break
                else:
                    candidates.append({"text": value, "models": [name]})

            if not candidates:
                stats["no_translation"] += 1
                continue
            seen.add(key)
            row = [text, code, "1"]
            row += [c["text"] for c in candidates] + [""] * (3 - len(candidates))
            row += [";".join(c["models"]) for c in candidates] \
                + [""] * (3 - len(candidates))
            rows.append(row)
            stats["written"] += 1
            per_lang[code] = per_lang.get(code, 0) + 1

    out = f"{SCR}/out/ghana-sentences-shola.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "language", "priority", "option1", "option2",
                    "option3", "model1", "model2", "model3"])
        w.writerows(rows)

    print(f"wrote {out}")
    print(f"  {'rows in dataset':36} {len(df):6,}")
    for reason, n in sorted(excluded.items(), key=lambda kv: -kv[1]):
        print(f"  excluded, {reason:26} {n:6,}")
    for k, v in stats.items():
        print(f"  {k:36} {v:6,}")
    print("\n  per language:")
    for code, n in sorted(per_lang.items(), key=lambda kv: -kv[1]):
        print(f"    {code:9} {n:6,}")
    opts = [sum(1 for c in r[3:6] if c) for r in rows]
    from collections import Counter
    print("\n  options per item:", dict(Counter(opts)))


if __name__ == "__main__":
    main()
