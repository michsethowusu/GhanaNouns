#!/usr/bin/env python3
"""
Extract all unique noun phrases (including multi-word nouns)
from a CSV file (column: 'text') using SpaCy.
Strips leading stop words from noun phrases before storage.
Only keeps noun phrases that are all lowercase.
Outputs a deduplicated CSV with phrase text, POS type ('NOUN_PHRASE'), and count.
"""
import os
import pandas as pd
from tqdm import tqdm
import spacy

# ------------------------------------------------------------------
# USER SETTINGS
# ------------------------------------------------------------------
INPUT_CSV   = "/content/drive/MyDrive/Collab/GhanaNouns/combined_sentences.csv"
TEXT_COLUMN = "text"          # column that holds the sentences
OUTPUT_FILE = "/content/drive/MyDrive/Collab/GhanaNouns/noun_phrases_news.csv"
BATCH_SIZE  = 10000
SAVE_EVERY  = 50000

# ------------------------------------------------------------------
# Load SpaCy model
print("🔤 Loading SpaCy model ('en_core_web_sm')...")
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

# Get SpaCy's stop words
stop_words = nlp.Defaults.stop_words
print(f"📋 Loaded {len(stop_words)} stop words from SpaCy")

# ------------------------------------------------------------------
# Load sentences from CSV
# ------------------------------------------------------------------
print(f"📂 Loading sentences from: {INPUT_CSV}")
df_in = pd.read_csv(INPUT_CSV)

if TEXT_COLUMN not in df_in.columns:
    raise KeyError(f"Column '{TEXT_COLUMN}' not found in {INPUT_CSV}")

sentences = df_in[TEXT_COLUMN].dropna().astype(str).str.strip()
sentences = sentences[sentences != ""].tolist()
print(f"📄 Loaded {len(sentences)} sentences")

# ------------------------------------------------------------------
# Resume logic
# ------------------------------------------------------------------
if os.path.exists(OUTPUT_FILE):
    processed_df = pd.read_csv(OUTPUT_FILE)
    noun_dict = {(row["phrase"].lower()): (row["phrase"], row["count"])
                 for _, row in processed_df.iterrows()
                 if pd.notna(row["phrase"]) and isinstance(row["phrase"], str)}
    print(f"🔁 Resuming from {len(noun_dict)} extracted noun phrases.")
else:
    noun_dict = {}

# ------------------------------------------------------------------
# Helper function to strip leading stop words
# ------------------------------------------------------------------
def strip_leading_stopwords(phrase, stop_words):
    """Remove stop words from the beginning of a phrase."""
    words = phrase.split()
    while words and words[0].lower() in stop_words:
        words.pop(0)
    return " ".join(words)

# ------------------------------------------------------------------
# Process sentences in batches
# ------------------------------------------------------------------
print("🚀 Starting noun phrase extraction...")
for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc="Processing"):
    batch = sentences[i:i + BATCH_SIZE]
    docs = list(nlp.pipe(batch, disable=["ner"]))

    for doc in docs:
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if phrase:
                # Strip leading stop words
                phrase = strip_leading_stopwords(phrase, stop_words)

                # Only store if there's still content after stripping
                # AND if the phrase is all lowercase
                if phrase and phrase == phrase.lower():
                    key = phrase.lower()
                    if key in noun_dict:
                        orig, count = noun_dict[key]
                        # Prefer the version with uppercase letters (but now we only keep lowercase)
                        if phrase != key and orig == key:
                            # New phrase has uppercase, old one doesn't
                            orig = phrase
                        noun_dict[key] = (orig, count + 1)
                    else:
                        noun_dict[key] = (phrase, 1)

    # Periodic checkpoint
    if (i + BATCH_SIZE) % SAVE_EVERY < BATCH_SIZE:
        temp_df = pd.DataFrame([
            {"phrase": orig, "pos": "NOUN_PHRASE", "count": c}
            for (_, (orig, c)) in noun_dict.items()
        ]).sort_values("count", ascending=False)
        temp_df.to_csv(OUTPUT_FILE, index=False)
        print(f"💾 Checkpoint saved ({len(noun_dict)} unique noun phrases).")

# ------------------------------------------------------------------
# Final save
# ------------------------------------------------------------------
final_df = (pd.DataFrame([
                {"phrase": orig, "pos": "NOUN_PHRASE", "count": c}
                for (_, (orig, c)) in noun_dict.items()
            ])
            .sort_values("count", ascending=False)
            .reset_index(drop=True))

final_df.to_csv(OUTPUT_FILE, index=False)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print(f"\n✅ Done! Extracted {len(final_df)} unique noun phrases.")
print("🔝 Top 10 most frequent noun phrases:")
print(final_df.head(10)[["phrase", "count"]].to_string(index=False))
