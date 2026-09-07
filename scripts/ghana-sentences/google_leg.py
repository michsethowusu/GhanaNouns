"""Google Translate free endpoint legs for Twi, Ewe and Ga.

Two translations per sentence:
  direct  source -> English
  pivot   source -> Thai -> English

Resumable: appends to out/google.jsonl keyed by "<config>:<index>".
"""
import json, os, random, sys, threading, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor

SCR = os.environ.get("SCR", "work")
OUT = f"{SCR}/out/google.jsonl"
ENDPOINT = "https://clients5.google.com/translate_a/t"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
URL_BUDGET = 1500          # keep the encoded query well under any URL limit
WORKERS = int(os.environ.get("WORKERS", "4"))

SRC = {"twi-aku": "ak", "twi-asa": "ak", "ewe": "ee", "gaa": "gaa"}

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


def flatten(node, out):
    """clients5 nests differently for one vs many q values."""
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for x in node:
            flatten(x, out)
    return out


def translate(texts, sl, tl, tries=6):
    """Return one English/Thai string per input, order preserved."""
    params = [("client", "dict-chrome-ex"), ("sl", sl), ("tl", tl)]
    params += [("q", t) for t in texts]
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode())
            got = flatten(d, [])
            if len(got) == len(texts):
                return got
            last = f"got {len(got)} for {len(texts)}"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except Exception as e:
            last = repr(e)
        time.sleep(min(90, 3 * 2 ** attempt) + random.random() * 4)
    if len(texts) == 1:
        raise RuntimeError(f"gave up: {last}")
    mid = len(texts) // 2                      # bisect on a length mismatch
    return (translate(texts[:mid], sl, tl, tries)
            + translate(texts[mid:], sl, tl, tries))


def batches(pend):
    """Group rows so the encoded URL stays inside URL_BUDGET."""
    cur, size = [], 0
    for i, t in pend:
        cost = len(urllib.parse.quote(t)) + 3
        if cur and (size + cost > URL_BUDGET or len(cur) >= 10):
            yield cur
            cur, size = [], 0
        cur.append((i, t))
        size += cost
    if cur:
        yield cur


def run_batch(job):
    global done_n
    config, chunk = job
    sl = SRC[config]
    texts = [t for _, t in chunk]
    try:
        direct = translate(texts, sl, "en")
        time.sleep(0.4)
        thai = translate(texts, sl, "th")
        time.sleep(0.4)
        pivot = translate(thai, "th", "en")
    except Exception as e:
        sys.stderr.write(f"batch {config} failed: {e}\n")
        return
    lines = [json.dumps({"uid": f"{config}:{i}", "config": config, "text": t,
                         "direct": d, "thai": th, "pivot": p},
                        ensure_ascii=False)
             for (i, t), d, th, p in zip(chunk, direct, thai, pivot)]
    with lock:
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        done_n += len(lines)
        if done_n % 500 < len(lines):
            print(f"  google {done_n} new rows", flush=True)
    time.sleep(0.5)


def main():
    import pandas as pd
    os.makedirs(f"{SCR}/out", exist_ok=True)
    df = pd.read_parquet(f"{SCR}/gs/all.parquet")
    df = df[df.config.isin(SRC)]
    seen = load_done()
    print(f"already done: {len(seen)}", flush=True)
    jobs = []
    for config, sub in df.groupby("config", sort=True):
        pend = [(i, t) for i, t in enumerate(sub.text.tolist())
                if f"{config}:{i}" not in seen]
        jobs += [(config, c) for c in batches(pend)]
    random.shuffle(jobs)
    print(f"batches to run: {len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(run_batch, jobs))
    print(f"google leg finished, {done_n} new rows", flush=True)


if __name__ == "__main__":
    main()
