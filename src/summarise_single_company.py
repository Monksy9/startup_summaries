import src.summarise_single_document as ssd

def summarise_top_articles(company_df, language_model, token_limit, topic, top_n_articles):
    """
    Summarise the top articles for a company based on the number of slashes in the URL.
    
    Parameters:
    - company_df (DataFrame): A dataframe containing company data.
    - language_model: The language model to use for summarisation.
    - token_limit (int): Token limit for the language model summarisation.
    - topic (str): The topic to summarise.

    Returns:
    - list: A list of summaries for the top articles.
    """
    top_articles_df = company_df.sort_values('slash_count').head(top_n_articles)
    
    summaries = [
        ssd.summarise_single_document(
            language_model, 
            token_limit, 
            topic, 
            article_text
        )
        for article_text in top_articles_df['text']
    ]
    
    return summaries
