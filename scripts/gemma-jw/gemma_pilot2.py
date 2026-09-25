"""Does Gemma 4 31B actually know these languages?

    .venv/bin/python gemma_pilot2.py

Translates the same 40 common words into every candidate language and measures
three things that distinguish knowing a language from producing text in its
general direction:

  echo      the output is the English word unchanged. Gemma does this when it
            has nothing; for a genuine loanword it is also correct, which is
            why a high rate is the signal, not any single word.
  twin      the output is near-identical to another language's. Related
            languages do share vocabulary, but a model that has never seen
            Ewe and is asked for it tends to return whatever it returns for
            the nearest language it does know.
  unparsed  no JSON array came back at all.

Known-good controls are included so the thresholds are calibrated against
languages Gemma demonstrably handles rather than against a guess.
"""
import json, os, re, sys, time
from pathlib import Path

HERE = Path("/mnt/volume_d2wey28/projects/shola-translate")
MODEL = "/mnt/volume_d2wey28/models/gemma-4-31B-it"
LANGS_TSV = HERE / "jw-new-langs.tsv"
REPORT = HERE / "pilot-jw.json"
os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf-cache")

# Everyday, concrete, and unlikely to be loanwords in most languages - so an
# echo really does mean the model had nothing.
WORDS = ["water", "child", "house", "fire", "road", "mother", "food", "night",
         "hand", "tree", "river", "moon", "blood", "salt", "goat", "market",
         "rain", "stone", "eye", "name", "friend", "money", "bird", "fish",
         "book", "sun", "door", "farm", "village", "song", "head", "wind",
         "milk", "egg", "snake", "cloth", "knife", "bone", "smoke", "star"]

# Gemma already produced usable output for these in the 55-language run.
CONTROLS = {"swa": "Swahili", "yor": "Yoruba", "hau": "Hausa",
            "zul": "Zulu", "amh": "Amharic", "twi": "Asante Twi"}


def parse(text, n):
    m = re.search(r"\[.*?\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return [str(x).strip() for x in arr] if isinstance(arr, list) and len(arr) == n else None


def main():
    langs = {}
    for line in LANGS_TSV.read_text(encoding="utf-8").splitlines():
        if line.strip():
            code, name = line.split("\t", 1)
            langs[code] = name
    langs.update(CONTROLS)
    print(f"{len(langs)} languages ({len(CONTROLS)} of them controls)", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=4096, trust_remote_code=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.2, max_tokens=len(WORDS) * 55)

    numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(WORDS))
    order, prompts = [], []
    for code, name in langs.items():
        user = (f"Translate each English word below into {name}.\n\nRules:\n"
                f"- Give the everyday word a speaker would actually use.\n"
                f"- Keep the tone marks and special letters the language is "
                f"written with.\n"
                f"- Return ONLY a JSON array of {len(WORDS)} strings, in the "
                f"same order, with nothing else.\n\n{numbered}")
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))
        order.append(code)

    t0 = time.time()
    outs = llm.generate(prompts, params, use_tqdm=True)
    print(f"generated in {time.time()-t0:.0f}s", flush=True)

    result = {}
    for code, out in zip(order, outs):
        got = parse(out.outputs[0].text, len(WORDS))
        if got is None:
            result[code] = {"name": langs[code], "unparsed": True}
            continue
        echo = sum(1 for w, t in zip(WORDS, got) if t.casefold() == w.casefold())
        blank = sum(1 for t in got if not t)
        result[code] = {"name": langs[code], "unparsed": False,
                        "echo": echo, "blank": blank,
                        "words": got}

    # Twins: two languages whose 40 answers agree almost entirely.
    keys = [c for c in result if not result[c]["unparsed"]]
    for c in keys:
        result[c]["twins"] = []
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            same = sum(1 for x, y in zip(result[a]["words"], result[b]["words"])
                       if x.casefold() == y.casefold())
            if same >= 36:                      # 90% of 40
                result[a]["twins"].append(b)
                result[b]["twins"].append(a)

    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    ctrl = [result[c] for c in CONTROLS if c in result and not result[c]["unparsed"]]
    if ctrl:
        print("\ncontrols (languages Gemma is known to handle):")
        for c in CONTROLS:
            r = result.get(c)
            if r and not r["unparsed"]:
                print(f"  {r['name']:16s} echo {r['echo']:>2}/40  "
                      f"twins {len(r['twins'])}")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
