"""Does Gemma give the same answer twice?

    .venv/bin/python gemma_consist.py

The echo test catches a model that gives up and hands back the English. It
cannot catch the failure that matters here: a model with no training data for
a language producing confident, plausible-looking invented words. Those score
zero echoes and are worthless.

Sampling the same prompt three times at a temperature that lets it wander
separates the two. A word the model actually knows comes back the same each
time. A word it is inventing does not.

Controls calibrate the threshold: whatever agreement Swahili and Yoruba show
is what knowing a language looks like on this measure.
"""
import json, os, re, time
from pathlib import Path

HERE = Path("/mnt/volume_d2wey28/projects/shola-translate")
MODEL = "/mnt/volume_d2wey28/models/gemma-4-31B-it"
LANGS_TSV = HERE / "jw-pass-langs.tsv"
REPORT = HERE / "consistency-jw2.json"
os.environ.setdefault("HF_HOME", "/mnt/volume_d2wey28/hf-cache")

WORDS = ["water", "child", "house", "fire", "road", "mother", "food", "night",
         "hand", "tree", "river", "moon", "blood", "salt", "goat", "market",
         "rain", "stone", "eye", "name"]

CONTROLS = {"swa": "Swahili", "yor": "Yoruba", "hau": "Hausa",
            "zul": "Zulu", "lin": "Lingala", "sna": "Shona"}


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
            c, n = line.split("\t", 1)
            langs[c] = n
    langs.update(CONTROLS)
    print(f"{len(langs)} languages x 3 samples", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=4096, trust_remote_code=True)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.8, top_p=0.95, n=3,
                            max_tokens=len(WORDS) * 55)

    numbered = "\n".join(f"{i+1}. {w}" for i, w in enumerate(WORDS))
    order, prompts = [], []
    for code, name in langs.items():
        user = (f"Translate each English word below into {name}.\n\nRules:\n"
                f"- Give the everyday word a speaker would actually use.\n"
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
        samples = [parse(o.text, len(WORDS)) for o in out.outputs]
        samples = [s for s in samples if s]
        if len(samples) < 2:
            result[code] = {"name": langs[code], "agree": None}
            continue
        agree = 0
        for i in range(len(WORDS)):
            vals = {s[i].casefold() for s in samples}
            if len(vals) == 1:
                agree += 1
        result[code] = {"name": langs[code], "samples": len(samples),
                        "agree": agree, "of": len(WORDS),
                        "all": samples}

    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print("\ncontrols:")
    for c in CONTROLS:
        r = result.get(c)
        if r and r["agree"] is not None:
            print(f"  {r['name']:14s} {r['agree']:>2}/{r['of']} stable   "
                  f"'water' -> {r['example']}")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
