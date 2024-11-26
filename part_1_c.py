import re

def custom_tokenizer(text):
    """
    Custom tokenizer that splits a tweet into tokens based on defined rules.
    """
    # Regex patterns
    mention_pattern = r"@\w+"           # Mentions
    hashtag_pattern = r"#\w+"           # Hashtags
    emoji_pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"  # Emojis
    word_pattern = r"[a-zA-Z0-9]+(?:'[a-z]+)?"  # Words and contractions
    punctuation_pattern = r"\.\.\.|[.,!?;:]"    # Ellipses and punctuation
    symbol_pattern = r"&\w+"                    # Symbols (e.g., &amp;)

    # Combine patterns into one regex
    combined_pattern = f"({mention_pattern}|{hashtag_pattern}|{emoji_pattern}|{word_pattern}|{punctuation_pattern}|{symbol_pattern})"

    # Find all tokens in the order they appear
    tokens = [match.group() for match in re.finditer(combined_pattern, text)]

    return tokens

def part_1_c(tweets_df):
    """
    Tokenizes the tweets in the 'text' column using the following rules:
    1. Extract mentions (e.g., @username).
    2. Extract hashtags (e.g., #topic).
    3. Extract emojis (using Unicode ranges for emojis).
    4. Extract words, including contractions (e.g., "flying", "didn't", "2023").
    5. Extract punctuation, including ellipses (e.g., "...", "!", ".").
    6. Extract symbols, such as HTML escape sequences (e.g., "&amp;").
    """

    # Apply custom tokenizer to the 'text' column
    tweets_df['tokens'] = tweets_df['text'].apply(custom_tokenizer)

    # Print sample tokenized tweets
    print(tweets_df[['text', 'tokens']].head(10))
