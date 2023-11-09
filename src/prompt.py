def extract_html_information_prompt(text: str, topic: str) -> str:
    """
    Generate a prompt for extracting and summarizing company information from HTML content.
    
    Parameters:
    text (str): The HTML text content to analyze.
    topic (str): The specific topic to focus the summary on.
    
    Returns:
    str: A formatted prompt for the AI to generate a summary.
    """
    prompt = (
        f"[INST]\n"
        "You are an AI designed to extract and summarize the purpose of a company from web page content. When presented with HTML content, your task is to analyze the text, "
        "images, and other data to identify and articulate the company's purpose. This may include the company's goals, product details, services, size, or any other business-related "
        "aspect that defines its mission and objectives.\n\n"
        f"Please review the provided HTML text, but do not visit any links, and summarize the {topic}, considering the content provided:\n\n"
        f"{text}\n\n"
        "Note: If the HTML content consists solely of error messages, system-generated responses, terms of service pages, disclaimers, or other automated content that does not provide "
        f"specific information about the {topic}, indicate that the {topic} could not be determined from the provided content.\n\n"
        f"Based on the content provided, summarize the {topic} in a brief paragraph.\n"
        "[/INST]"
    )
    return prompt

def summary_prompt(text: str, topic: str) -> str:
    """
    Generate a prompt for summarizing key information from a set of summaries related to a company's web content.
    
    Parameters:
    text (str): The text containing individual summaries to be consolidated.
    topic (str): The specific topic to focus the consolidated summary on.
    
    Returns:
    str: A formatted prompt for the AI to generate a consolidated summary.
    """
    prompt = (
        f"[INST]\n"
        "You are an AI designed to extract and summarize the purpose of a company from web page content. If you are not confident there is sufficient content in the summaries, e.g., "
        "it appears to be HTML noise for unrelated pages, say that.\n\n"
        f"Please review the following summaries of web pages, and provide a single summary containing key information, and summarize the {topic}\n"
        f"{text}\n"
        "CONCISE summary:\n\n"
        "[/INST]"
    )
    return prompt
