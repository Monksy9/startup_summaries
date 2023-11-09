from langdetect import detect
from typing import Optional

def is_english(text: str) -> bool:
    """
    Determine whether the given text is in English.

    Parameters:
    text (str): The text to analyze for language detection.

    Returns:
    bool: True if the text is detected to be in English, False otherwise.
    """
    try:
        language: Optional[str] = detect(text)
    except Exception: 
        return False
    return language == 'en'
