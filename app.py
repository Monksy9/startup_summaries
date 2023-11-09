import os
import pandas as pd
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from typing import List, Dict  
from src import summarise_single_company as ssc
from src import summarise_single_document as ssd
from src import preprocess as pp

REPO_ID = "mistralai/Mistral-7B-Instruct-v0.1"
TOKEN_ENV_VAR = 'HUGGINGFACEHUB_API_TOKEN'
MODEL_KWARGS = {'temperature': 0.2, 'top_p': 0.7, 'top_k': 55}
FILE_PATH = 'investor_interest_topics.txt'
TOKEN_LIMIT_MISTRAL_7B = 1024
PATH_TO_CSV = 'task_sources.csv'
OUTPUT_FILE = 'summaries.csv'
TOP_N_ARTICLES = 5
SUBSAMPLE_COMPANIES = 3

def load_model() -> HuggingFaceHub:
    """Load the HuggingFaceHub model with the required API token and model parameters."""
    api_token = os.getenv(TOKEN_ENV_VAR)
    if api_token is None:
        raise ValueError("API token is not set in the environment variables.")
    return HuggingFaceHub(repo_id=REPO_ID, huggingfacehub_api_token=api_token, model_kwargs=MODEL_KWARGS)


def read_investor_interest_topics(file_path: str) -> List[str]:
    """Read the investor interest topics from a given file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]


def summarise_companies(df: pd.DataFrame, llm: HuggingFaceHub, token_limit: int, investor_interest_topics: List[str]) -> pd.DataFrame:
    """Summarise information for each company, for each topic in the dataframe using the provided language model."""
    companies = df['second_level_domain'].unique()
    summaries = pd.DataFrame()
    for company in companies[0:SUBSAMPLE_COMPANIES + 1]:  
        print(f"Summarising for company: {company}")
        for topic in investor_interest_topics:
            print(f"Summarising for topic: {topic}")
            filtered_df = df[df['second_level_domain'] == company]
            company_summaries = ssc.summarise_top_articles(filtered_df, llm, token_limit, topic, top_n_articles=TOP_N_ARTICLES)
            summaries.loc[company, topic] = ssd.summarise_summaries(company_summaries, llm, topic)
    return summaries


def main():
    """Main function to load the credentials, model, process data, and summarise companies."""
    load_dotenv()
    
    llm = load_model()
    investor_interest_topics = read_investor_interest_topics(FILE_PATH)
    processed_data = pp.preprocess_and_load_data(PATH_TO_CSV)
    company_summaries = summarise_companies(processed_data, llm, TOKEN_LIMIT_MISTRAL_7B, investor_interest_topics)
    company_summaries.to_csv(OUTPUT_FILE)


if __name__ == "__main__":
    main()
