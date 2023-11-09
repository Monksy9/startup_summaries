import re
from typing import List

def detect_error(message: str) -> bool:
    """
    Detect if a given message contains any of the predefined error indicators.

    Parameters:
    message (str): The message string to be checked for error indicators.

    Returns:
    bool: True if an error indicator is found, False otherwise.
    """
    
    error_indicators: List[str] = [
        r"Error:",
        r"Timeout \d+ms exceeded",
        r"Target closed"
    ]
    
    error_pattern = re.compile("|".join(error_indicators))
    match = error_pattern.search(message)
    return bool(match)
