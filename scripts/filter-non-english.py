import os
import pandas as pd
import fasttext
import requests
import wget
from tqdm import tqdm

# ===== CONFIGURATION – EDIT THESE =====
input_csv = "combined_phrases_with_percentages.csv"          # path to your 1M‑row CSV
text_column = "phrase"              # column containing phrases
output_csv = "combined_phrases_with_percentages_filtered.csv"
confidence_threshold = 0.7          # recommended: 0.7–0.8 for balance; 0.9+ for strict
chunk_size = 10000                 # rows per chunk (adjust for RAM)
model_variant = "lid.176.bin"      # use .bin for max accuracy, .ftz for smaller/faster
cache_dir = "./fasttext_models"    # where to store the downloaded model
# ======================================

# ---- Download model automatically (if not already cached) ----
os.makedirs(cache_dir, exist_ok=True)
model_path = os.path.join(cache_dir, model_variant)

FASTTEXT_URLS = {
    "lid.176.bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
}

if not os.path.exists(model_path):
    print(f"Downloading {model_variant} from Facebook AI... (~126 MB)")
    url = FASTTEXT_URLS[model_variant]
    wget.download(url, out=model_path)
    print("\nDownload complete.")
else:
    print(f"Model found at {model_path}")

# ---- Load model (once, reused for all chunks) ----
print("Loading fastText model...")
ft_model = fasttext.load_model(model_path)
print("Model ready.\n")

# ---- Process CSV in chunks ----
first_chunk = True
total_rows = 0
kept_rows = 0

# Get total lines for progress bar (fast)
total_lines = sum(1 for _ in open(input_csv, 'r', encoding='utf-8'))

with tqdm(total=total_lines, desc="Processing rows", unit=" rows") as pbar:
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        # --- Language identification + scoring ---
        def classify(text):
            if pd.isna(text) or not isinstance(text, str):
                return None, 0.0
            # FastText expects newlines removed
            clean = text.strip().replace("\n", " ")
            predictions = ft_model.predict(clean)   # returns (labels, probabilities)
            lang = predictions[0][0].replace("__label__", "")
            score = predictions[1][0]
            return lang, score

        lang_scores = chunk[text_column].apply(lambda x: classify(x))
        chunk["lang_code"] = lang_scores.apply(lambda x: x[0])
        chunk["lang_score"] = lang_scores.apply(lambda x: x[1])

        # --- Filter: English + confidence ≥ threshold ---
        mask = (chunk["lang_code"] == "en") & (chunk["lang_score"] >= confidence_threshold)
        filtered = chunk[mask].drop(columns=["lang_code", "lang_score"])  # optional: keep if you want

        # --- Write output ---
        if first_chunk:
            filtered.to_csv(output_csv, index=False)
            first_chunk = False
        else:
            filtered.to_csv(output_csv, mode='a', header=False, index=False)

        # --- Stats ---
        kept_rows += len(filtered)
        total_rows += len(chunk)
        pbar.update(len(chunk))

print("\n" + "="*60)
print(f"COMPLETE")
print(f"  Total rows processed : {total_rows:,}")
print(f"  English rows kept    : {kept_rows:,} ({kept_rows/total_rows*100:.1f}%)")
print(f"  Confidence threshold : {confidence_threshold}")
print(f"  Output file          : {output_csv}")
print("="*60)
