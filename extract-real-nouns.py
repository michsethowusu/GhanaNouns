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
# Extraction code
# ------------------------------------------------------------

import pandas as pd
from openai import OpenAI
import json
from tqdm import tqdm
import time
from collections import defaultdict
import os

# Mistral API Configuration
MISTRAL_API_KEY = "your_mistral_api_key_here"  # Replace with your Mistral API key

# Initialize OpenAI client for Mistral
client = OpenAI(
    base_url="https://api.mistral.ai/v1",
    api_key=MISTRAL_API_KEY
)

MODEL = "mistral-large-latest"  # Using Mistral Large model
BATCH_SIZE = 100  

# Resume functionality
RESUME_STATE_FILE = 'classification_resume_state.json'
PROGRESS_FILE = 'classification_progress.csv'

def save_resume_state(batch_num, word_classifications, total_batches, input_file, output_file, text_column):
    """Save the current state for resume capability using JSON"""
    state = {
        'batch_num': batch_num,
        'word_classifications': word_classifications,
        'total_batches': total_batches,
        'input_file': input_file,
        'output_file': output_file,
        'text_column': text_column,
        'timestamp': time.time(),
        'batch_size': BATCH_SIZE
    }
    try:
        with open(RESUME_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"  💾 Saved resume state (completed batch {batch_num}/{total_batches})")
        return True
    except Exception as e:
        print(f"  ⚠️  Error saving resume state: {e}")
        return False

def load_resume_state():
    """Load the saved state if it exists"""
    if os.path.exists(RESUME_STATE_FILE):
        try:
            with open(RESUME_STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            print(f"  📂 Found resume state - completed batch {state['batch_num']}/{state['total_batches']}")
            return state
        except Exception as e:
            print(f"  ⚠️  Error loading resume state: {e}")
            return None
    return None

def clear_resume_state():
    """Clear the resume state files"""
    if os.path.exists(RESUME_STATE_FILE):
        os.remove(RESUME_STATE_FILE)
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    print("  🗑️  Cleared resume state")

def classify_batch_concrete_nouns(words):
    """
    Send up to 200 unique words to Mistral and get JSON response.
    Returns list of dicts with word and classification.
    """
    
    # Prepare the word list for the prompt
    word_list = "\n".join([f"{i+1}. {word}" for i, word in enumerate(words)])
    
    prompt = f"""Classify each word/phrase into one of three categories: CONCRETE, ABSTRACT, or NON-NOUN.

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

Return all {len(words)} words in the exact order given. No explanations, just the JSON array."""

    try:
        print(f"  → Sending request to {MODEL}...")
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            top_p=0.95,
            max_tokens=len(words) * 40,
            stream=False
        )
        
        print(f"  → Received response")
        content = completion.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                content = content[start_idx:end_idx]
            
            classifications = json.loads(content)
            
            if isinstance(classifications, list) and len(classifications) == len(words):
                return classifications
            else:
                print(f"  ⚠ Warning: Expected {len(words)} results, got {len(classifications) if isinstance(classifications, list) else 'invalid'}")
                return None
            
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parsing error: {e}")
            print(f"  Raw response (first 300 chars): {content[:300]}")
            return None
        
    except Exception as e:
        print(f"  ✗ API error: {type(e).__name__}: {e}")
        return None

def get_unique_words_with_mapping(df, text_column):
    """
    Extract unique words from dataframe and create mapping.
    Returns: (unique_words_list, word_to_indices_dict)
    """
    texts = df[text_column].astype(str).str.strip().tolist()
    
    word_to_indices = defaultdict(list)
    for idx, word in enumerate(texts):
        word_to_indices[word].append(idx)
    
    unique_words = list(word_to_indices.keys())
    
    print(f"Total rows: {len(texts)}")
    print(f"Unique words: {len(unique_words)}")
    print(f"Duplicates removed: {len(texts) - len(unique_words)} ({(1 - len(unique_words)/len(texts))*100:.1f}% reduction)")
    
    return unique_words, word_to_indices

def process_csv_in_batches(input_file, output_file, text_column='text', use_thinking=False, filter_non_nouns=True, resume=True):
    """
    Process CSV file in batches of 200 unique words using Mistral with resume capability.
    """
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"\n{'='*60}")
    print(f"Original rows: {len(df)}")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Thinking mode: {'Enabled' if use_thinking else 'Disabled (faster)'}")
    print(f"Filter non-nouns: {'Yes' if filter_non_nouns else 'No'}")
    print(f"Resume enabled: {'Yes' if resume else 'No'}")
    print(f"{'='*60}\n")
    
    # Get unique words and mapping
    unique_words, word_to_indices = get_unique_words_with_mapping(df, text_column)
    
    # Calculate total batches
    total_batches = (len(unique_words) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Check for resume state
    start_batch = 0
    word_classifications = {}
    
    if resume:
        resume_state = load_resume_state()
        if (resume_state and 
            resume_state.get('input_file') == input_file and 
            resume_state.get('output_file') == output_file and
            resume_state.get('batch_size') == BATCH_SIZE):
            
            completed_batch = resume_state['batch_num']
            if completed_batch < total_batches:
                start_batch = completed_batch
                word_classifications = resume_state['word_classifications']
                print(f"🔄 Resuming from batch {start_batch + 1}/{total_batches}")
                print(f"  📊 Previously processed: {len(word_classifications)} words")
                print(f"  🎯 Remaining batches: {total_batches - start_batch}")
            else:
                print("  ✅ All batches completed in previous run")
                start_batch = total_batches
        else:
            print("  🆕 Starting fresh (no valid resume state)")
    
    # Process unique words in batches
    print(f"\nProcessing {len(unique_words)} unique words in {total_batches} batches of {BATCH_SIZE}...\n")
    
    successful_batches = 0
    failed_batches = 0
    
    classify_func = classify_batch_concrete_nouns
    
    try:
        for i in tqdm(range(start_batch * BATCH_SIZE, len(unique_words), BATCH_SIZE), 
                     desc="Processing", initial=start_batch, total=total_batches):
            
            batch = unique_words[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} words...")
            
            # Classify batch
            classifications = classify_func(batch)
            
            if classifications and isinstance(classifications, list):
                # Store classifications
                classified_count = 0
                for classification in classifications:
                    if isinstance(classification, dict) and 'word' in classification and 'label' in classification:
                        word = classification['word']
                        label = classification['label']
                        word_classifications[word] = label
                        classified_count += 1
                
                print(f"  ✓ Successfully classified {classified_count}/{len(batch)} words")
                successful_batches += 1
            else:
                print(f"  ✗ Failed to classify batch {batch_num}")
                failed_batches += 1
                # Mark failed words as UNKNOWN
                for word in batch:
                    word_classifications[word] = 'UNKNOWN'
            
            # Save resume state after each batch
            if resume:
                save_resume_state(batch_num, word_classifications, total_batches, input_file, output_file, text_column)
            
            # Rate limiting
            if i + BATCH_SIZE < len(unique_words):
                time.sleep(0.5)  # Reduced wait time for larger batches
        
        # Clear resume state on successful completion
        if resume and start_batch < total_batches:
            clear_resume_state()
            print("  ✅ Processing completed successfully - resume state cleared")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Process interrupted by user!")
        print(f"  💾 Progress saved - resume with the same command")
        return None, None
    except Exception as e:
        print(f"\n\n❌ Error during processing: {e}")
        print(f"  💾 Progress saved - resume with the same command")
        return None, None
    
    # Apply classifications to all rows
    print(f"\n{'='*60}")
    print(f"Applying classifications to all {len(df)} rows...")
    print(f"{'='*60}\n")
    
    labels = ['UNKNOWN'] * len(df)
    for word, indices in word_to_indices.items():
        label = word_classifications.get(word, 'UNKNOWN')
        for idx in indices:
            labels[idx] = label
    
    df['classification'] = labels
    
    # Filter results
    if filter_non_nouns:
        concrete_df = df[df['classification'] == 'CONCRETE'].copy()
    else:
        concrete_df = df[df['classification'].isin(['CONCRETE', 'ABSTRACT'])].copy()
    
    concrete_df = concrete_df.drop('classification', axis=1)
    
    # Save results
    concrete_df.to_csv(output_file, index=False)
    
    # Also save full results with classifications
    full_output = output_file.replace('.csv', '_with_labels.csv')
    df.to_csv(full_output, index=False)
    
    # Statistics
    concrete_count = sum(df['classification'] == 'CONCRETE')
    abstract_count = sum(df['classification'] == 'ABSTRACT')
    non_noun_count = sum(df['classification'] == 'NON-NOUN')
    unknown_count = sum(df['classification'] == 'UNKNOWN')
    
    # Show examples
    print("\n" + "="*60)
    print("SAMPLE CLASSIFICATIONS:")
    print("="*60)
    
    sample_concrete = df[df['classification'] == 'CONCRETE'][text_column].head(10).tolist()
    sample_abstract = df[df['classification'] == 'ABSTRACT'][text_column].head(10).tolist()
    sample_non_noun = df[df['classification'] == 'NON-NOUN'][text_column].head(10).tolist()
    
    if sample_concrete:
        print("\n✓ CONCRETE examples:")
        for idx, word in enumerate(sample_concrete, 1):
            print(f"  {idx}. {word}")
    
    if sample_abstract:
        print("\n✗ ABSTRACT examples:")
        for idx, word in enumerate(sample_abstract, 1):
            print(f"  {idx}. {word}")
    
    if sample_non_noun:
        print("\n⚠ NON-NOUN examples:")
        for idx, word in enumerate(sample_non_noun, 1):
            print(f"  {idx}. {word}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Total rows:               {len(df)}")
    print(f"Unique words processed:   {len(unique_words)}")
    print(f"API calls made:           {total_batches}")
    print(f"Successful batches:       {successful_batches}")
    print(f"Failed batches:           {failed_batches}")
    print(f"\nClassifications:")
    print(f"  CONCRETE:               {concrete_count} ({concrete_count/len(df)*100:.1f}%)")
    print(f"  ABSTRACT:               {abstract_count} ({abstract_count/len(df)*100:.1f}%)")
    print(f"  NON-NOUN:               {non_noun_count} ({non_noun_count/len(df)*100:.1f}%)")
    print(f"  UNKNOWN:                {unknown_count} ({unknown_count/len(df)*100:.1f}%)")
    print(f"\nOutput files:")
    print(f"  → {output_file}")
    print(f"  → {full_output}")
    print(f"{'='*60}\n")
    
    return concrete_df, df

def test_api_connection():
    """Test the Mistral API connection"""
    print("="*60)
    print("Testing Mistral API connection...")
    print("="*60 + "\n")
    
    if MISTRAL_API_KEY == "your_mistral_api_key_here":
        print("✗ Error: API key not set!")
        return False
    
    print(f"API Key (first 10 chars): {MISTRAL_API_KEY[:10]}...")
    print(f"API URL: {client.base_url}")
    print(f"Model: {MODEL}\n")
    
    test_words = ["dog", "happiness", "run", "beautiful", "table", "freedom"]
    
    print(f"Testing with {len(test_words)} sample words...\n")
    
    result = classify_batch_concrete_nouns(test_words)
    
    if result:
        print("\n✓ API connection successful!\n")
        print("Sample classifications:")
        print("-" * 50)
        for item in result:
            if item['label'] == 'CONCRETE':
                symbol = "✓"
            elif item['label'] == 'ABSTRACT':
                symbol = "✗"
            elif item['label'] == 'NON-NOUN':
                symbol = "⚠"
            else:
                symbol = "?"
            print(f"  {symbol} {item['word']:15} → {item['label']}")
        print("-" * 50)
        return True
    else:
        print("\n✗ API connection failed")
        print("\nTroubleshooting:")
        print("1. Verify your API key at: https://console.mistral.ai/")
        print("2. Check your internet connection")
        print("3. Try: pip install --upgrade openai")
        return False

# Main execution
if __name__ == "__main__":
    if MISTRAL_API_KEY == "your_mistral_api_key_here":
        print("\n" + "="*60)
        print("⚠️  MISTRAL API KEY REQUIRED")
        print("="*60)
        print("\nSteps to get your API key:")
        print("1. Go to: https://console.mistral.ai/")
        print("2. Sign up or log in")
        print("3. Navigate to API Keys section")
        print("4. Generate a new API key")
        print("5. Copy and paste it into this script")
        print("\nNote: Mistral offers free trial credits!")
        print("="*60 + "\n")
    else:
        # Test connection
        if test_api_connection():
            print("\n" + "="*60)
            print("Starting full CSV processing...")
            print("="*60 + "\n")
            time.sleep(2)
            
            # Process CSV
            concrete_df, full_df = process_csv_in_batches(
                input_file='input.csv',
                output_file='concrete_nouns_only.csv',
                text_column='text',  # Change to match your column name
                use_thinking=False,  # Set to True to see reasoning
                filter_non_nouns=True,  # Set to False to keep abstract nouns
                resume=True  # Set to False to disable resume
            )
