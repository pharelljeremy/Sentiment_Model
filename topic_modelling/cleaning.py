import pandas as pd
import re
from langdetect import detect, DetectorFactory
import nltk
from nltk.corpus import stopwords
import os

# Fix seed for langdetect to make language detection deterministic
DetectorFactory.seed = 0

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Keywords related to WWDC and Apple products/events
KEYWORDS = [
    'wwdc', 'apple', 'ios', 'macos', 'iphone', 'ipad', 'macbook', 'watchos', 'tvos',
    'airpods', 'applewatch', 'safari', 'swift', 'xcode', 'appstore', 'apple silicon'
]

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def clean_text(text):
    if not isinstance(text, str):
        return ''
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\S+', '', text)
    # Remove non-alpha characters (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def filter_and_clean_chunk(chunk, keywords_pattern):
    # Filter rows where 'combined_text' contains any keyword (case insensitive)
    mask = chunk['combined_text'].str.contains(keywords_pattern, case=False, na=False)
    filtered = chunk.loc[mask].copy()

    if filtered.empty:
        return pd.DataFrame()  # Empty dataframe

    # Clean text
    filtered['clean_text'] = filtered['combined_text'].apply(clean_text)

    # Remove rows with empty cleaned text or very short text (<10 chars)
    filtered = filtered[filtered['clean_text'].str.len() >= 10]

    # Keep only English rows
    filtered = filtered[filtered['clean_text'].apply(is_english)]

    # Remove stopwords
    filtered['clean_text'] = filtered['clean_text'].apply(remove_stopwords)

    return filtered

def clean_large_csv(input_csv, output_csv, chunksize=100000):
    first_chunk = True
    total_rows = 0
    total_kept = 0

    # Try to get total rows for progress estimation
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f) - 1  # subtract header
    except Exception:
        total_lines = None

    keywords_pattern = '|'.join(KEYWORDS)

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, dtype=str):
        total_rows += len(chunk)

        cleaned_chunk = filter_and_clean_chunk(chunk, keywords_pattern)
        if cleaned_chunk.empty:
            # Still print progress
            if total_lines:
                percent = (total_rows / total_lines) * 100
                print(f"Processed {total_rows} rows ({percent:.2f}%), kept {total_kept} rows so far...")
            else:
                print(f"Processed {total_rows} rows, kept {total_kept} rows so far...")
            continue

        if first_chunk:
            cleaned_chunk.to_csv(output_csv, index=False, mode='w')
            first_chunk = False
        else:
            cleaned_chunk.to_csv(output_csv, index=False, mode='a', header=False)

        total_kept += len(cleaned_chunk)

        if total_lines:
            percent = (total_rows / total_lines) * 100
            print(f"Processed {total_rows} rows ({percent:.2f}%), kept {total_kept} rows so far...")
        else:
            print(f"Processed {total_rows} rows, kept {total_kept} rows so far...")

    print(f"Finished cleaning. Total rows processed: {total_rows}, total rows kept: {total_kept}")

if __name__ == "__main__":
    # Set your file paths here:
    input_file = r"/Users/jeremypharell/Desktop/£/School/Year 3/ITDPA/topic_modelling/csv's/combine_datasets.csv"
    output_file = r"/Users/jeremypharell/Desktop/£/School/Year 3/ITDPA/topic_modelling/csv's/cleaned_combine_datasets.csv"

    clean_large_csv(input_file, output_file)

