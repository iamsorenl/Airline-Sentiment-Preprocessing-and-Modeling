import re

def part_1_c(tweets_df):
    """
    Tokenizes the tweets in the 'text' column using the following rules:
    1. Extract mentions (e.g., @username).
    2. Extract hashtags (e.g., #topic).
    3. Extract emojis (using Unicode ranges for emojis).
    4. Extract words and numbers (e.g., "flying", "2023").
    5. Extract punctuation (e.g., "!", ".").
    """

    def custom_tokenizer(text):
        """
        Custom tokenizer that splits a tweet into tokens based on defined rules.
        """
        # Rule 1: Match mentions (e.g., @username)
        mentions = re.findall(r"@\w+", text)
        
        # Rule 2: Match hashtags (e.g., #topic)
        hashtags = re.findall(r"#\w+", text)
        
        # Rule 3: Match emojis (basic Unicode range for emojis)
        emojis = re.findall(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", text)
        
        # Rule 4: Match words and numbers (e.g., "flying", "2023")
        words = re.findall(r"\b\w+\b", text)
        
        # Rule 5: Match punctuation (e.g., "!")
        punctuation = re.findall(r"[^\w\s]", text)
        
        # Combine all tokens
        tokens = mentions + hashtags + emojis + words + punctuation
        return tokens

    # Apply custom tokenizer to the 'text' column
    tweets_df['tokens'] = tweets_df['text'].apply(custom_tokenizer)

    # Print sample tokenized tweets
    print(tweets_df[['text', 'tokens']].head(10))
