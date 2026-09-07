"""Translate ghana-sentences `text` to English with Gemini 3.6 Flash.

Resumable: results append to out/gemini.jsonl keyed by "<config>:<index>".
"""
import json, os, random, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

SCR = os.environ.get("SCR", "work")
KEY = os.environ["GEMINI_KEY"]
MODEL = "gemini-3.6-flash"
URL = (f"https://generativelanguage.googleapis.com/v1beta/models/"
       f"{MODEL}:generateContent?key={KEY}")
OUT = f"{SCR}/out/gemini.jsonl"
BATCH = 12
WORKERS = int(os.environ.get("WORKERS", "8"))

LANG = {"ada": "Dangme", "dag": "Dagbani", "dga": "Dagaare", "ewe": "Ewe",
        "fat": "Fante", "gaa": "Ga", "gjn": "Gonja", "gur": "Gurene",
        "nzi": "Nzema", "twi-aku": "Akuapem Twi", "twi-asa": "Asante Twi",
        "xsm": "Kasem"}

lock = threading.Lock()
done_n = 0


def load_done():
    seen = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line)["uid"])
                except Exception:
                    pass
    return seen


def prompt_for(language, texts):
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return (
        f"You are a professional {language} to English translator. {language} is "
        f"a Ghanaian language.\n\n"
        f"Translate each numbered {language} line below into natural English.\n\n"
        f"Rules:\n"
        f"- Translate meaning, not word for word.\n"
        f"- Some lines are fragments, headings or partial sentences from school "
        f"books. Translate the fragment as it stands; do not complete it or add "
        f"anything.\n"
        f"- If a line is a personal name, a page number or is not {language} at "
        f"all, return it unchanged.\n"
        f"- Return ONLY a JSON array of {len(texts)} strings, in the same order, "
        f"with no other text.\n\n"
        f"{numbered}"
    )


def call(payload_text, tries=6):
    body = json.dumps({
        "contents": [{"parts": [{"text": payload_text}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": "low"},
        },
    }).encode()
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
            cand = d["candidates"][0]
            parts = cand.get("content", {}).get("parts") or []
            txt = "".join(p.get("text", "") for p in parts)
            if not txt.strip():
                raise ValueError(f"empty text, finish={cand.get('finishReason')}")
            return txt
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt) + random.random() * 3)
                continue
            raise
        except Exception as e:                       # timeouts, empty, parse
            last = repr(e)
            time.sleep(min(30, 2 ** attempt) + random.random() * 2)
    raise RuntimeError(f"gave up: {last}")


def parse(txt, n):
    try:
        arr = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", txt, re.S)
        if not m:
            return None
        try:
            arr = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if isinstance(arr, dict):                        # occasional {"translations": [...]}
        for v in arr.values():
            if isinstance(v, list):
                arr = v
                break
    if not isinstance(arr, list) or len(arr) != n:
        return None
    return ["" if x is None else str(x).strip() for x in arr]


def run_batch(job):
    global done_n
    config, chunk = job
    texts = [t for _, t in chunk]
    try:
        txt = call(prompt_for(LANG[config], texts))
        out = parse(txt, len(texts))
        if out is None:                              # split and retry one by one
            out = []
            for t in texts:
                single = call(prompt_for(LANG[config], [t]))
                one = parse(single, 1)
                out.append(one[0] if one else "")
    except Exception as e:
        sys.stderr.write(f"batch {config} failed: {e}\n")
        return
    lines = [json.dumps({"uid": f"{config}:{i}", "config": config,
                         "text": t, "en": en}, ensure_ascii=False)
             for (i, t), en in zip(chunk, out)]
    with lock:
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        done_n += len(lines)
        if done_n % 600 < len(lines):
            print(f"  gemini {done_n} new rows", flush=True)


def main():
    import pandas as pd
    os.makedirs(f"{SCR}/out", exist_ok=True)
    df = pd.read_parquet(f"{SCR}/gs/all.parquet")
    seen = load_done()
    print(f"already done: {len(seen)}", flush=True)
    jobs = []
    for config, sub in df.groupby("config", sort=True):
        pend = [(i, t) for i, t in enumerate(sub.text.tolist())
                if f"{config}:{i}" not in seen]
        for k in range(0, len(pend), BATCH):
            jobs.append((config, pend[k:k + BATCH]))
    random.shuffle(jobs)
    print(f"batches to run: {len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(run_batch, jobs))
    print(f"gemini leg finished, {done_n} new rows", flush=True)


if __name__ == "__main__":
    main()
