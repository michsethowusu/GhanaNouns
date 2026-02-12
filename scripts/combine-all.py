import pandas as pd
import spacy
from tqdm import tqdm
import re

def has_multiple_capitalized_words(phrase):
    """Check if more than one word starts with a capital letter"""
    if pd.isna(phrase):
        return False
    words = str(phrase).split()
    capitalized_count = sum(1 for word in words if word and word[0].isupper())
    return capitalized_count > 1

def is_all_caps(phrase):
    """Check if all alphabetic words are fully capitalized"""
    if pd.isna(phrase):
        return False
    words = str(phrase).split()
    alphabetic_words = [word for word in words if any(c.isalpha() for c in word)]
    return len(alphabetic_words) > 0 and all(word.isupper() for word in alphabetic_words)

def should_remove(phrase):
    """Remove if all caps OR more than one capitalized word"""
    return is_all_caps(phrase) or has_multiple_capitalized_words(phrase)

def contains_non_alphabetic_chars(phrase):
    """
    Check if phrase contains non-alphabetic characters.
    Allows letters, spaces, and hyphens (but not if hyphen is at start/end or surrounded by spaces).
    Numbers are considered non-alphabetic and will trigger removal.
    """
    if pd.isna(phrase):
        return True
    
    phrase = str(phrase)
    
    # Check for numbers anywhere
    if any(char.isdigit() for char in phrase):
        return True
    
    # Check each character
    for i, char in enumerate(phrase):
        if char.isalpha() or char.isspace():
            continue
        
        if char == '-':
            # Check if hyphen is at start, end, or surrounded by spaces
            if i == 0 or i == len(phrase) - 1:  # At start or end
                return True
            if phrase[i-1].isspace() or phrase[i+1].isspace():  # Surrounded by space
                return True
            # Valid hyphen (between two non-space characters)
            continue
        else:
            # Any other non-alphabetic character
            return True
    
    return False

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])  # Disable unused components for speed

def batch_contains_adjective(phrases, batch_size=1000):
    """Process phrases in batches with progress bar"""
    results = []
    for doc in tqdm(nlp.pipe(phrases, batch_size=batch_size), total=len(phrases), desc="Checking adjectives"):
        has_adj = any(token.pos_ == "ADJ" for token in doc)
        results.append(has_adj)
    return results

# Read CSVs with updated filenames
print("Reading CSVs...")
news_df = pd.read_csv('noun_phrases_news.csv')
research_df = pd.read_csv('noun_phrases_research.csv')

# Verify columns exist
print(f"News columns: {news_df.columns.tolist()}")
print(f"Research columns: {research_df.columns.tolist()}")

# Drop NaN phrases
news_df = news_df.dropna(subset=['phrase'])
research_df = research_df.dropna(subset=['phrase'])

print(f"Original - News: {len(news_df)}, Research: {len(research_df)}")

# Step 1: Remove phrases with non-alphabetic characters
print("Removing phrases with non-alphabetic characters from news...")
news_df = news_df[~news_df['phrase'].apply(contains_non_alphabetic_chars)]

print("Removing phrases with non-alphabetic characters from research...")
research_df = research_df[~research_df['phrase'].apply(contains_non_alphabetic_chars)]

print(f"After non-alphabetic removal - News: {len(news_df)}, Research: {len(research_df)}")

# Step 2: Remove all-caps and multi-capitalized phrases
print("Removing all-caps and multi-capitalized phrases from news...")
news_df = news_df[~news_df['phrase'].apply(should_remove)]

print("Removing all-caps and multi-capitalized phrases from research...")
research_df = research_df[~research_df['phrase'].apply(should_remove)]

print(f"After capitalized removal - News: {len(news_df)}, Research: {len(research_df)}")

# Step 3: Filter adjectives using batch processing with progress bar
print("Filtering adjectives from news...")
news_df['has_adj'] = batch_contains_adjective(news_df['phrase'].tolist(), batch_size=2000)
news_df = news_df[~news_df['has_adj']].drop(columns=['has_adj'])

print("Filtering adjectives from research...")
research_df['has_adj'] = batch_contains_adjective(research_df['phrase'].tolist(), batch_size=2000)
research_df = research_df[~research_df['has_adj']].drop(columns=['has_adj'])

print(f"After adjective filtering - News: {len(news_df)}, Research: {len(research_df)}")

# Calculate totals
total_news = news_df['count'].sum()
total_research = research_df['count'].sum()
print(f"Total news: {total_news}, Total research: {total_research}")

# Create lowercase keys
news_df['key'] = news_df['phrase'].str.lower()
research_df['key'] = research_df['phrase'].str.lower()

# Merge
merged = pd.merge(
    news_df[['key', 'phrase', 'count']].rename(columns={'count': 'news_count', 'phrase': 'news_phrase'}),
    research_df[['key', 'phrase', 'count']].rename(columns={'count': 'research_count', 'phrase': 'research_phrase'}),
    on='key',
    how='outer'
)

merged['news_count'] = merged['news_count'].fillna(0)
merged['research_count'] = merged['research_count'].fillna(0)
merged['phrase'] = merged['news_phrase'].fillna(merged['research_phrase'])

# Calculate percentages
merged['news_percentage'] = (merged['news_count'] / total_news * 100).round(4)
merged['research_percentage'] = (merged['research_count'] / total_research * 100).round(4)

merged['average_percentage'] = merged.apply(
    lambda row: ((row['news_percentage'] + row['research_percentage']) / 2) 
    if row['news_count'] > 0 and row['research_count'] > 0 
    else (row['news_percentage'] if row['news_count'] > 0 else row['research_percentage']),
    axis=1
).round(4)

merged['source'] = merged.apply(
    lambda row: 'both' if row['news_count'] > 0 and row['research_count'] > 0 
    else ('news' if row['news_count'] > 0 else 'research'),
    axis=1
)

result = merged[['phrase', 'news_count', 'research_count', 'news_percentage', 
                 'research_percentage', 'average_percentage', 'source']]
result = result.sort_values('average_percentage', ascending=False)

result.to_csv('combined_phrases_with_percentages.csv', index=False)
print(f"Done! {len(result)} unique phrases")
print(result['source'].value_counts())
