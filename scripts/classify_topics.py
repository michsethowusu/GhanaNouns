import os
import pandas as pd
import time
import re
from pathlib import Path
from openai import OpenAI

# Initialize the Mistral API client
client = OpenAI(
    base_url="https://api.mistral.ai/v1",
    api_key=os.environ.get("MISTRAL_API_KEY")
)

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
INPUT_FILE  = "/home/owusus/Documents/GitHub/GhanaNouns/data/ghana-nouns.csv"           # ← change this
OUTPUT_FILE = "/home/owusus/Documents/GitHub/GhanaNouns/data/ghana-nouns_classified.csv"  # ← change this
PHRASE_COLUMN = "phrase"   # Set to column name string, or None to auto-detect (first column)
BATCH_SIZE = 100
SLEEP_BETWEEN_BATCHES = 2   # seconds
# ──────────────────────────────────────────────────────────────────────────────

TOPICS = [
    "Agriculture and Food Production",
    "Healthcare and Wellbeing",
    "General",
    "None of the above",
]

TOPIC_KEYS = [
    "agriculture_and_food_production",
    "healthcare_and_wellbeing",
    "general",
    "none_of_the_above",
]

# Map display names → column keys
TOPIC_MAP = dict(zip(TOPICS, TOPIC_KEYS))


CLASSIFY_PROMPT = """You are a topic classifier. Classify each noun phrase into ONE OR MORE of these topics:

Topics:
- Agriculture and Food Production  (farming, crops, livestock, food systems, soil, irrigation, harvest, agribusiness, food security, nutrition, diet, food processing)
- Healthcare and Wellbeing  (medicine, disease, mental health, hospitals, drugs, vaccines, fitness, public health, patient care, medical research, wellness)
- General  (broad societal, economic, political, technological, cultural, or educational topics that don't fit the above)
- None of the above  (proper nouns with no clear topical meaning, highly ambiguous fragments, or junk phrases)

Rules:
1. A phrase CAN belong to multiple topics.
2. Use "None of the above" ONLY when genuinely unclassifiable or meaningless.
3. Output ONLY the XML block below — no explanation, no extra text.
4. There must be EXACTLY {n} <item> blocks, one per phrase, in the same order.
5. Each <item> uses id= matching the phrase number, and exactly these 4 child tags with values 1 (true) or 0 (false):
   <agri>, <health>, <general>, <none>

Example for 2 phrases:
<results>
  <item id="1"><agri>1</agri><health>0</health><general>0</general><none>0</none></item>
  <item id="2"><agri>0</agri><health>1</health><general>1</general><none>0</none></item>
</results>

Phrases to classify:
{phrases_text}"""


def _call_api(prompt: str) -> str:
    """Single API call, returns stripped response text."""
    completion = client.chat.completions.create(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=8192,
        stream=False
    )
    return completion.choices[0].message.content.strip()


def _parse_results(response_text: str, expected: int) -> list[dict] | None:
    """
    Parse XML response into a list of normalised dicts.
    Returns None if the item count doesn't match expected.

    Parsing is tag-based (not position-based), so extra whitespace,
    comments, or reordered attributes don't break it.
    """
    # Extract all <item ...>...</item> blocks
    items = re.findall(r'<item[^>]*>(.*?)</item>', response_text, re.DOTALL)

    if len(items) != expected:
        return None

    def extract(block: str, tag: str) -> bool:
        m = re.search(rf'<{tag}>\s*([01])\s*</{tag}>', block)
        return bool(int(m.group(1))) if m else False

    results = []
    for block in items:
        results.append({
            "agriculture_and_food_production": extract(block, "agri"),
            "healthcare_and_wellbeing":        extract(block, "health"),
            "general":                         extract(block, "general"),
            "none_of_the_above":               extract(block, "none"),
        })
    return results


def _classify_one(phrase: str) -> dict:
    """Fallback: classify a single phrase. Always returns a valid dict."""
    prompt = CLASSIFY_PROMPT.format(n=1, phrases_text=f"1. {phrase}")
    try:
        text = _call_api(prompt)
        result = _parse_results(text, 1)
        if result:
            return result[0]
    except Exception:
        pass
    return {k: False for k in TOPIC_KEYS}


def classify_batch(phrases: list[str], max_retries: int = 2) -> list[dict]:
    """
    Classify a batch of noun phrases into one or more topics.

    Self-healing strategy:
      1. Try the full batch (up to max_retries times).
      2. If the count still doesn't match after retries, split into two halves
         and recurse — this isolates whichever half is confusing the model.
      3. If a batch reaches size 1 and still fails, classify that single phrase
         with a dedicated single-item call (_classify_one).

    This guarantees len(output) == len(input) with real classifications,
    never silent padding with False.
    """
    n = len(phrases)

    if n == 0:
        return []

    # ── Single-phrase fast path ───────────────────────────────────────────────
    if n == 1:
        return [_classify_one(phrases[0])]

    phrases_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(phrases)])
    prompt = CLASSIFY_PROMPT.format(n=n, phrases_text=phrases_text)

    # ── Retry loop ────────────────────────────────────────────────────────────
    for attempt in range(1, max_retries + 1):
        try:
            text = _call_api(prompt)
            result = _parse_results(text, n)
            if result is not None:
                return result
            print(f"  ⚠ Attempt {attempt}/{max_retries}: count mismatch on batch of {n}")
        except Exception as e:
            print(f"  ✗ Attempt {attempt}/{max_retries}: API error — {e}")
        if attempt < max_retries:
            time.sleep(1)

    # ── Divide-and-conquer fallback ───────────────────────────────────────────
    mid = n // 2
    print(f"  ↪ Splitting batch of {n} → [{mid}, {n - mid}] for independent retry")
    left  = classify_batch(phrases[:mid],  max_retries)
    right = classify_batch(phrases[mid:],  max_retries)
    return left + right


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    # ── Load input ──────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found!")
        return

    print(f"Loaded {len(df):,} rows from {INPUT_FILE}")
    print(f"Columns: {df.columns.tolist()}")

    phrase_col = PHRASE_COLUMN if PHRASE_COLUMN else df.columns[0]
    print(f"Using phrase column: '{phrase_col}'")

    # ── Add topic columns if missing ─────────────────────────────────────────
    for col in TOPIC_KEYS:
        if col not in df.columns:
            df[col] = pd.NA   # nullable — NA means not yet processed

    # ── Resume logic ─────────────────────────────────────────────────────────
    start_idx = 0
    if Path(OUTPUT_FILE).exists():
        print(f"\nFound existing output file — resuming…")
        existing = pd.read_csv(OUTPUT_FILE, low_memory=False)
        if all(k in existing.columns for k in TOPIC_KEYS):
            # Count rows where ALL topic columns are non-null (i.e. processed)
            processed_mask = existing[TOPIC_KEYS].notna().all(axis=1)
            start_idx = int(processed_mask.sum())
            df = existing
            print(f"Already processed: {start_idx:,} rows — resuming from index {start_idx}")
        else:
            print("Topic columns missing in existing file — starting fresh")
    else:
        print("No existing output file — starting fresh")

    # ── Process batches ──────────────────────────────────────────────────────
    remaining = len(df) - start_idx
    total_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nProcessing {remaining:,} rows in {total_batches} batches (size={BATCH_SIZE})")
    print("=" * 70)

    process_start = time.time()

    for batch_num in range(total_batches):
        batch_start = start_idx + batch_num * BATCH_SIZE
        batch_end   = min(batch_start + BATCH_SIZE, len(df))
        phrases     = df.loc[batch_start : batch_end - 1, phrase_col].tolist()

        pct = (batch_num / total_batches * 100) if total_batches > 0 else 100
        print(f"\n[Batch {batch_num+1}/{total_batches}] [{pct:.1f}%] rows {batch_start}–{batch_end-1}")

        classifications = classify_batch(phrases)

        for i, cls in enumerate(classifications):
            row_idx = batch_start + i
            for col, val in cls.items():
                df.loc[row_idx, col] = val

        # Save after every batch
        df.to_csv(OUTPUT_FILE, index=False)

        elapsed  = time.time() - process_start
        avg      = elapsed / (batch_num + 1)
        eta      = avg * (total_batches - batch_num - 1)

        # Quick summary of this batch's results
        batch_df = df.loc[batch_start : batch_end - 1, TOPIC_KEYS]
        counts   = {k: int(batch_df[k].sum()) for k in TOPIC_KEYS}
        print(f"  ✓ agri={counts['agriculture_and_food_production']}  "
              f"health={counts['healthcare_and_wellbeing']}  "
              f"general={counts['general']}  "
              f"none={counts['none_of_the_above']}")
        print(f"  ✓ Elapsed: {format_duration(elapsed)} | ETA: {format_duration(eta)} | Avg: {avg:.1f}s/batch")

        if batch_num < total_batches - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - process_start
    print(f"\n{'='*70}")
    print(f"✓ Classification complete!  Total time: {format_duration(total_elapsed)}")
    print(f"✓ Results saved to: {OUTPUT_FILE}")
    print(f"\nLabel distribution (rows can have multiple labels):")
    for col, label in zip(TOPIC_KEYS, TOPICS):
        n = int(df[col].sum())
        pct = n / len(df) * 100
        print(f"  {label:<40} {n:>7,}  ({pct:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
