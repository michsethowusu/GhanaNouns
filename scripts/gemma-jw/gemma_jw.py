"""Translate the SHOLA word list into the JW African languages Gemma can handle.

    .venv/bin/python gemma_jw.py

Reads `jw-final-langs.tsv`, the languages that survived gemma_pilot2.py, and
works tier by tier so tier 1 - the only tier anyone is being asked about - is
finished and publishable while the long tail is still running.

Resumable: a checkpoint records how many chunks of each (tier, language) are
done, so an interrupted run skips them without re-reading the output. That
matters here because this is tens of hours.
"""
import json, os, re, signal, sys, time
from pathlib import Path

HERE = Path("/mnt/volume_d2wey28/projects/shola-translate")
MODEL = "/mnt/volume_d2wey28/models/gemma-4-31B-it"
OUT = HERE / "out"
CKPT = HERE / "checkpoint-jw.json"
LANGS_TSV = HERE / "jw-final-langs.tsv"
CHUNK = 40
STEP = 200
os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf-cache")

stopping = False


def stop(signum, frame):
    global stopping
    stopping = True
    print("\n[stopping after this batch - checkpoint is safe]", flush=True)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def load_words():
    by_tier = {}
    for line in (HERE / "words-kept.tsv").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        phrase, tier = line.rsplit("\t", 1)
        by_tier.setdefault(int(tier), []).append(phrase)
    return by_tier


def chunks(words):
    for i in range(0, len(words), CHUNK):
        yield words[i:i + CHUNK]


def parse(text, n):
    m = re.search(r"\[.*?\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return arr if isinstance(arr, list) and len(arr) == n else None


def main():
    OUT.mkdir(exist_ok=True)
    langs = {}
    for line in LANGS_TSV.read_text(encoding="utf-8").splitlines():
        if line.strip():
            code, name = line.split("\t", 1)
            langs[code] = name
    done = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    by_tier = load_words()
    total = sum(len(v) for v in by_tier.values()) * len(langs)
    print(f"{total:,} translations across {len(langs)} languages", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=4096, trust_remote_code=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.2, max_tokens=CHUNK * 55)

    started, written, bad = time.time(), 0, 0
    for tier in sorted(by_tier):
        words = by_tier[tier]
        path = OUT / f"gemma-jw-tier{tier}.jsonl"
        for code, name in langs.items():
            key = f"{tier}:{code}"
            offset = done.get(key, 0)
            todo = list(chunks(words))[offset:]
            if not todo:
                continue
            prompts = []
            for chunk in todo:
                numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(chunk))
                user = (f"Translate each English word or phrase below into "
                        f"{name}.\n\nRules:\n"
                        f"- Give the everyday word a speaker would actually "
                        f"use.\n"
                        f"- Keep the tone marks and special letters the "
                        f"language is written with.\n"
                        f"- Return ONLY a JSON array of {len(chunk)} strings, "
                        f"in the same order, with nothing else.\n\n{numbered}")
                prompts.append(tok.apply_chat_template(
                    [{"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True))

            for s in range(0, len(prompts), STEP):
                if stopping:
                    print(f"stopped: {written:,} written this run", flush=True)
                    return
                outs = llm.generate(prompts[s:s + STEP], params, use_tqdm=False)
                lines = []
                for chunk, out in zip(todo[s:s + STEP], outs):
                    got = parse(out.outputs[0].text, len(chunk))
                    if got is None:
                        bad += 1
                        continue
                    for w, t in zip(chunk, got):
                        t = str(t).strip()
                        # Identical to the English tells a speaker nothing.
                        if t and t.casefold() != w.casefold():
                            lines.append(json.dumps(
                                {"phrase": w, "language": code, "text": t},
                                ensure_ascii=False))
                if lines:
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write("\n".join(lines) + "\n")
                    written += len(lines)
                done[key] = offset + min(s + STEP, len(prompts))
                CKPT.write_text(json.dumps(done))
            rate = written / max(1, time.time() - started)
            print(f"[tier {tier}] {name} ({code}) done | {written:,} written "
                  f"| {rate:.0f}/s", flush=True)
        print(f"=== tier {tier} complete -> {path}", flush=True)

    print(f"\nDONE: {written:,} written, {bad} unparseable batches", flush=True)


if __name__ == "__main__":
    main()
