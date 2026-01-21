# ------------------------------------------------------------
# Auto-install missing requirements
# ------------------------------------------------------------
import sys, subprocess, importlib

pkgs = {
    "pandas": "pandas>=1.5.0",
    "openai": "openai>=1.0.0",
    "tqdm": "tqdm>=4.64.0"
}
for import_name, pip_spec in pkgs.items():
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"📦 {import_name} not found → installing {pip_spec} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec])
# ------------------------------------------------------------
# Normal imports
# ------------------------------------------------------------
import pandas as pd
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIError
import json
from tqdm import tqdm
import time
from collections import defaultdict
import os

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
MISTRAL_API_KEY = "your_mistral_api_key_here"  # ← put key here
client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=MISTRAL_API_KEY)
MODEL = "mistral-large-latest"
BATCH_SIZE = 200  # keep as you wish
RESUME_STATE_FILE = "classification_resume_state.json"

# -----------------------------------------------------------
# RETRY CONFIG
# -----------------------------------------------------------
MAX_NETWORK_RETRIES = 100  # Maximum retries for network errors (practically infinite)
MAX_CONTENT_RETRIES = 3    # Maximum retries for content/parsing errors
INITIAL_RETRY_DELAY = 1    # Initial retry delay in seconds
MAX_RETRY_DELAY = 300      # Maximum retry delay in seconds (5 minutes)
BACKOFF_FACTOR = 2         # Exponential backoff factor

# -----------------------------------------------------------
# UTILS
# -----------------------------------------------------------
def save_resume_state(batch_num, word_classifications, total_batches, input_file, output_file, text_column):
    state = {
        "batch_num": batch_num,
        "word_classifications": word_classifications,
        "total_batches": total_batches,
        "input_file": input_file,
        "output_file": output_file,
        "text_column": text_column,
        "timestamp": time.time(),
        "batch_size": BATCH_SIZE,
    }
    with open(RESUME_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved resume state (completed batch {batch_num}/{total_batches})")

def load_resume_state():
    if not os.path.exists(RESUME_STATE_FILE):
        return None
    try:
        with open(RESUME_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  Error loading resume state: {e}")
        return None

def clear_resume_state():
    if os.path.exists(RESUME_STATE_FILE):
        os.remove(RESUME_STATE_FILE)
    print("  🗑️  Cleared resume state")

def calculate_retry_delay(attempt, is_network_error=False):
    """Calculate exponential backoff delay with jitter"""
    if is_network_error:
        # Longer delays for network errors
        delay = min(INITIAL_RETRY_DELAY * (BACKOFF_FACTOR ** (attempt - 1)), MAX_RETRY_DELAY)
    else:
        # Shorter delays for content errors
        delay = min(INITIAL_RETRY_DELAY * (BACKOFF_FACTOR ** (attempt - 1)), 30)
    
    # Add jitter (±10%)
    jitter = delay * 0.1
    delay += (jitter * (2 * (time.time() % 1) - 1))
    return max(1, delay)  # Minimum 1 second

def safe_sleep(seconds, interrupt_check_interval=5):
    """Sleep with interrupt checking"""
    end_time = time.time() + seconds
    while time.time() < end_time:
        remaining = end_time - time.time()
        sleep_time = min(interrupt_check_interval, remaining)
        if sleep_time <= 0:
            break
        time.sleep(sleep_time)

# -----------------------------------------------------------
# 1.  ENHANCED RETRY-AWARE CLASSIFIER
# -----------------------------------------------------------
def classify_batch_concrete_nouns(words):
    """
    Classify a batch of words with enhanced retry logic.
    Returns List[dict] on success, None on content failure after retries.
    """
    n = len(words)
    word_list = "\n".join([f"{i+1}. {w}" for i, w in enumerate(words)])

    prompts = [
        f"""Classify each word/phrase into one of three categories: CONCRETE, ABSTRACT, or NON-NOUN. in the SAME order and dont try to do any cleaning or splitting of the input words. You must classify them exactly as they are. DONT return more entries than you were given, even if one entry contains muliple words.
        
CONCRETE: Physical objects you can perceive with your senses (see, touch, smell, taste, hear)
Examples: dog, table, water, car, book, phone, building, apple

ABSTRACT: Concepts, ideas, emotions, qualities you cannot physically perceive
Examples: love, freedom, happiness, justice, democracy, time, theory, peace

NON-NOUN: Words that are not nouns at all (verbs, adjectives, adverbs, prepositions, etc.)
Examples: run (verb), beautiful (adjective), quickly (adverb), in (preposition), and (conjunction)

Words to classify:
{word_list}

Respond ONLY with a JSON array. Format:
[
  {{"word": "dog", "label": "CONCRETE"}},
  {{"word": "happiness", "label": "ABSTRACT"}},
  {{"word": "run", "label": "NON-NOUN"}}
]

Return exactly {n} words in the exact order given. No explanations, just the JSON array.""",
        f"""You MUST return a JSON array with EXACTLY {n} elements, one per word below, in the SAME order and dont try to do any cleaning or splitting of the input words. You must classify them exactly as they are.

Words:
{word_list}

Reply format (no other text):
[
  {{"word": "<exact_word>", "label": "CONCRETE"}},
  ...
]

Labels: CONCRETE, ABSTRACT, or NON-NOUN."""
    ]

    network_attempt = 0
    content_attempt = 0
    
    while True:
        # Select prompt (stricter on retry for content errors)
        prompt = prompts[1] if content_attempt > 0 else prompts[0]
        
        try:
            network_attempt += 1
            print(f"  → Sending request to {MODEL} (Network attempt {network_attempt})...")
            
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.95,
                max_tokens=n * 40,
                stream=False,
                timeout=120  # Increased timeout
            )

            content = completion.choices[0].message.content.strip()

            # --- JSON extraction -------------------------------------------------
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()

            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found")
            content = content[start_idx:end_idx]

            classifications = json.loads(content)
            
            # --- validation -------------------------------------------------------
            if not isinstance(classifications, list) or len(classifications) != n:
                raise ValueError(
                    f"Expected {n} results, got {len(classifications) if isinstance(classifications, list) else 'invalid'}"
                )
            
            # Validate each entry
            for i, cls in enumerate(classifications):
                if not isinstance(cls, dict) or 'word' not in cls or 'label' not in cls:
                    raise ValueError(f"Invalid entry at position {i}: {cls}")
                if cls['label'] not in ['CONCRETE', 'ABSTRACT', 'NON-NOUN']:
                    raise ValueError(f"Invalid label at position {i}: {cls['label']}")
            
            print(f"  ✓ Valid response")
            return classifications

        except (APITimeoutError, APIConnectionError, APIError) as e:
            # Network-level failure → retry indefinitely
            delay = calculate_retry_delay(network_attempt, is_network_error=True)
            print(f"  🔄 Network/timeout error: {e}")
            print(f"  ⏳ Retrying in {delay:.1f}s (Network attempt {network_attempt})...")
            safe_sleep(delay)
            continue  # Continue retrying indefinitely
            
        except (RateLimitError, json.JSONDecodeError, ValueError) as e:
            # Model replied but content is bad
            content_attempt += 1
            if content_attempt >= MAX_CONTENT_RETRIES:
                print(f"  ⚠️  Content issue after {MAX_CONTENT_RETRIES} attempts: {e}")
                print(f"  ⚠️  Classifying batch as UNKNOWN")
                return None
                
            delay = calculate_retry_delay(content_attempt, is_network_error=False)
            print(f"  🔄 Content issue on attempt {content_attempt}/{MAX_CONTENT_RETRIES}: {e}")
            print(f"  ⏳ Retrying in {delay:.1f}s...")
            safe_sleep(delay)
            
        except KeyboardInterrupt:
            raise  # Re-raise to be caught by outer handler
            
        except Exception as e:
            print(f"  ⚠️  Unexpected error: {e}")
            # For unexpected errors, retry with network error logic
            delay = calculate_retry_delay(network_attempt, is_network_error=True)
            print(f"  ⏳ Retrying in {delay:.1f}s...")
            safe_sleep(delay)

# -----------------------------------------------------------
# helper – build unique-word list + reverse index
# -----------------------------------------------------------
def get_unique_words_with_mapping(df, text_column):
    texts = df[text_column].astype(str).str.strip().tolist()
    word_to_indices = defaultdict(list)
    for idx, word in enumerate(texts):
        word_to_indices[word].append(idx)
    unique_words = list(word_to_indices.keys())
    print(f"Total rows: {len(texts)}")
    print(f"Unique words: {len(unique_words)}")
    print(f"Duplicates removed: {len(texts) - len(unique_words)} "
          f"({(1 - len(unique_words)/len(texts))*100:.1f}% reduction)")
    return unique_words, word_to_indices

# -----------------------------------------------------------
# 2.  BATCH PROCESSOR WITH ENHANCED RETRY
# -----------------------------------------------------------
def process_csv_in_batches(input_file, output_file, text_column='text',
                           use_thinking=False, filter_non_nouns=True, resume=True):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    print(f"\n{'='*60}")
    print(f"Original rows: {len(df)}")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Thinking mode: {'Enabled' if use_thinking else 'Disabled'}")
    print(f"Filter non-nouns: {'Yes' if filter_non_nouns else 'No'}")
    print(f"Resume enabled: {'Yes' if resume else 'No'}")
    print(f"{'='*60}\n")

    unique_words, word_to_indices = get_unique_words_with_mapping(df, text_column)
    total_batches = (len(unique_words) + BATCH_SIZE - 1) // BATCH_SIZE

    # resume logic
    start_batch, word_classifications = 0, {}
    if resume:
        state = load_resume_state()
        if (state and state.get('input_file') == input_file and
            state.get('output_file') == output_file and state.get('batch_size') == BATCH_SIZE):
            start_batch = state['batch_num']
            word_classifications = state['word_classifications']
            print(f"🔄 Resuming from batch {start_batch + 1}/{total_batches}")
        else:
            print("  🆕 Starting fresh (no valid resume state)")

    print(f"\nProcessing {len(unique_words)} unique words in {total_batches} batches of {BATCH_SIZE}...")
    print(f"Will retry indefinitely on network errors with exponential backoff up to {MAX_RETRY_DELAY}s\n")

    successful_batches = failed_batches = 0
    classify_func = classify_batch_concrete_nouns

    try:
        for i in tqdm(range(start_batch * BATCH_SIZE, len(unique_words), BATCH_SIZE),
                     desc="Processing", initial=start_batch, total=total_batches):

            batch = unique_words[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1

            print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} words...")
            print(f"  Unique words classified so far: {len(word_classifications)}/{len(unique_words)}")

            # Classify with enhanced retry logic
            classifications = classify_func(batch)

            if classifications:  # SUCCESS
                for cls in classifications:
                    if isinstance(cls, dict) and 'word' in cls and 'label' in cls:
                        word_classifications[cls['word']] = cls['label']
                print(f"  ✓ Successfully classified {len(classifications)}/{len(batch)} words")
                successful_batches += 1
            else:  # CONTENT failure after retries
                print(f"  ⚠️  Classifying {len(batch)} words as UNKNOWN after retries")
                for w in batch:
                    word_classifications[w] = 'UNKNOWN'
                failed_batches += 1

            # save state after every batch
            if resume:
                save_resume_state(batch_num, word_classifications,
                                total_batches, input_file, output_file, text_column)

            # Small delay between successful batches to avoid rate limits
            if i + BATCH_SIZE < len(unique_words):
                time.sleep(0.5)

        # finished
        if resume:
            clear_resume_state()
            print("  ✅ Processing completed successfully – resume state cleared")

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user!")
        print("  💾 Progress saved – resume with the same command")
        return None, None
    except Exception as e:
        print(f"\n\n❌ Error during processing: {e}")
        print("  💾 Progress saved – resume with the same command")
        return None, None

    # apply classifications & export
    print(f"\n{'='*60}")
    print(f"Applying classifications to all {len(df)} rows...")
    print(f"{'='*60}\n")

    labels = ['UNKNOWN'] * len(df)
    for word, indices in word_to_indices.items():
        label = word_classifications.get(word, 'UNKNOWN')
        for idx in indices:
            labels[idx] = label
    df['classification'] = labels

    if filter_non_nouns:
        out_df = df[df['classification'] == 'CONCRETE'].copy()
    else:
        out_df = df[df['classification'].isin(['CONCRETE', 'ABSTRACT'])].copy()
    out_df = out_df.drop(columns=['classification'])
    out_df.to_csv(output_file, index=False)

    full_out = output_file.replace('.csv', '_with_labels.csv')
    df.to_csv(full_out, index=False)

    # stats
    concrete = (df['classification'] == 'CONCRETE').sum()
    abstract = (df['classification'] == 'ABSTRACT').sum()
    non_noun = (df['classification'] == 'NON-NOUN').sum()
    unknown = (df['classification'] == 'UNKNOWN').sum()

    print(f"\n{'='*60}\nFINAL RESULTS:\n{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Total batches: {total_batches}  |  Successful: {successful_batches}  |  Failed: {failed_batches}")
    print(f"CONCRETE: {concrete}  |  ABSTRACT: {abstract}  |  NON-NOUN: {non_noun}  |  UNKNOWN: {unknown}")
    print(f"Output: {output_file}  |  Full labels: {full_out}")
    print(f"{'='*60}\n")
    return out_df, df


# -----------------------------------------------------------
# 3.  TEST + MAIN
# -----------------------------------------------------------
def test_api_connection():
    print("=" * 60)
    print("Testing Mistral API connection...")
    print("=" * 60 + "\n")
    if MISTRAL_API_KEY == "your_mistral_api_key_here":
        print("✗ Error: API key not set!")
        return False
    print(f"API Key (first 10 chars): {MISTRAL_API_KEY[:10]}...")
    print(f"API URL: {client.base_url}")
    print(f"Model: {MODEL}\n")
    test_words = ["dog", "happiness", "run", "beautiful", "table", "freedom"]
    result = classify_batch_concrete_nouns(test_words)
    if result:
        print("\n✓ API connection successful!\n")
        for item in result:
            sym = "✓" if item['label'] == 'CONCRETE' else "✗" if item['label'] == 'ABSTRACT' else "⚠"
            print(f"  {sym} {item['word']:15} → {item['label']}")
        return True
    else:
        print("\n✗ API connection test failed")
        print("1. Verify key at https://console.mistral.ai/")
        print("2. Check internet / firewall")
        print("3. pip install -U openai")
        return False


if __name__ == "__main__":
    if MISTRAL_API_KEY == "your_mistral_api_key_here":
        print("\n" + "=" * 60 + "\n⚠️  MISTRAL API KEY REQUIRED\n" + "=" * 60)
        print("1. Go to https://console.mistral.ai/\n2. Generate key\n3. Paste into script\n" + "=" * 60 + "\n")
    else:
        if test_api_connection():
            print("\n" + "=" * 60 + "\nStarting full CSV processing...\n" + "=" * 60 + "\n")
            time.sleep(2)
            process_csv_in_batches(
                input_file='input.csv',  # <-- change if needed
                output_file='concrete_nouns_only.csv',
                text_column='text',
                use_thinking=False,
                filter_non_nouns=True,
                resume=True
            )
