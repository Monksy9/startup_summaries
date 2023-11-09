from typing import List
from langchain import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
import src.prompt as p

def summarise_single_document(llm: HuggingFaceHub, token_limit: int, topic: str, text: str) -> str:
    """
    Generate a summary for a single document.

    Parameters:
    llm (HuggingFaceHub): A language model for generating summaries.
    token_limit (int): The token limit for the language model.
    topic (str): The topic based on which to generate the summary.
    text (str): The text of the document to summarize.

    Returns:
    str: The final summary of the document.
    """
    total_tokens_needed = token_calculation(llm, text)
    chunked_texts = chunk_documents(text, token_limit)
    summaries = summarise_chunks(chunked_texts, llm, topic)
    final_summary = summarise_summaries(summaries, llm, topic)
    return final_summary

def token_calculation(llm: HuggingFaceHub, text: str) -> int:
    """
    Calculate the number of tokens in the text using the language model.

    Parameters:
    llm (HuggingFaceHub): The language model used for token calculation.
    text (str): The text to calculate tokens for.

    Returns:
    int: The number of tokens in the text.
    """
    return llm.get_num_tokens(text)

def chunk_documents(text: str, token_limit: int) -> List[str]:
    """
    Chunk documents based on token limit.

    Parameters:
    text (str): The text to be chunked.
    token_limit (int): The token limit for each chunk.

    Returns:
    List[str]: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=token_limit)
    return text_splitter.create_documents([text])

def summarise_chunks(text_chunks: List[str], llm: HuggingFaceHub, topic: str) -> List[str]:
    """
    Summarise each chunk of text.

    Parameters:
    text_chunks (List[str]): The list of text chunks to summarize.
    llm (HuggingFaceHub): The language model used for summarization.
    topic (str): The topic based on which to generate the summary.

    Returns:
    List[str]: A list of summaries for each chunk.
    """
    summaries = []
    for chunk in text_chunks:
        prompt = p.extract_html_information_prompt(chunk, topic)
        summary = llm(prompt, max_new_tokens=2048, repetition_penalty=1.2)
        summaries.append(summary)
    return summaries

def summarise_summaries(summaries: List[str], llm: HuggingFaceHub, topic: str) -> str:
    """
    Combine individual summaries into a final summary.

    Parameters:
    summaries (List[str]): The list of individual summaries to combine.
    llm (HuggingFaceHub): The language model used for summarization.
    topic (str): The topic based on which to generate the summary.

    Returns:
    str: The final combined summary.
    """
    combined_summaries = '\n'.join(summaries)
    prompt = p.summary_prompt(combined_summaries, topic)
    final_summary = llm(prompt, max_new_tokens=2048, repetition_penalty=1.2)
    return final_summary
