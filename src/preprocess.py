from typing import List, Dict, Any
from urllib.parse import urlparse

import pandas as pd
import tldextract

from src import english as pp
from src import errors as e

def word_count(texts: List[str]) -> int:
    """Calculate the total number of words in a list of texts."""
    return len(' '.join(texts).split())

def log_text_length_for_each_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Log the text length and word count for each domain in the dataframe."""
    aggregation = df.groupby('second_level_domain')['text'].agg(['sum', 'count'])
    aggregation['text_length'] = aggregation['sum'].str.len()
    word_counts = df.groupby('second_level_domain')['text'].apply(word_count)
    aggregation['word_count'] = word_counts
    sorted_aggregation = aggregation.sort_values(by='text_length', ascending=False)

    for domain, row in sorted_aggregation.iterrows():
        print(f"Domain: {domain} | Text Length: {row['text_length']} | URL Count: {row['count']} | Word Count: {row['word_count']}")

    sorted_aggregation.rename(columns={'sum': 'text'}, inplace=True)
    return sorted_aggregation

def extract_main_domain(url: str) -> str:
    """Extract the main domain from a URL."""
    return tldextract.extract(url).domain

def to_sentence_case(text: str) -> str:
    """Convert text to sentence case."""
    return '. '.join(s.capitalize() for s in text.split('. '))

def is_base_url(url: str) -> bool:
    """Check if a URL is a base URL."""
    parsed_url = urlparse(url)
    return parsed_url.path in ('', '/') and not parsed_url.query and not parsed_url.fragment

def counter_slashes(url: str) -> int:
    """Count the number of slashes in a URL."""
    return url.count('/')

def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(path)

def preprocess_and_load_data(path: str) -> pd.DataFrame:
    """Preprocess data and load it from a CSV file."""
    df = load_data(path)
    print(f'Number of rows in original dataset: {df.shape[0]}')

    df['second_level_domain'] = df['url'].apply(extract_main_domain)
    df['is_english'] = df['text'].apply(pp.is_english)
    df['has_error'] = df['text'].apply(e.detect_error)
    df['base_url'] = df['url'].apply(is_base_url)
    df['slash_count'] = df['url'].apply(counter_slashes)
    df['text'] = df['text'].replace('\n', ' ').replace(r'\s+', ' ', regex=True).str.strip()
    df['text'] = df['text'].apply(to_sentence_case)

    filtered_df = df[df['is_english'] & ~df['has_error']]
    print(f'Number of rows in filtered dataset: {filtered_df.shape[0]}')

    log_text_length_for_each_domain(filtered_df)

    return filtered_df

def main() -> None:
    """Main function to preprocess and load data."""
    path_to_csv = 'task_sources.csv'
    processed_data = preprocess_and_load_data(path_to_csv)
    print(processed_data.head())

if __name__ == '__main__':
    main()
