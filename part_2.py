import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet') # Download the WordNet corpus

def clean_text(text):
    """
    Cleans a tweet text using the following steps:
    1. Remove mentions (e.g., @username).
    2. Remove currency symbols and amounts (e.g., "$19.90").
    3. Remove email addresses (e.g., "jane.doe@email.com").
    4. Remove emojis (e.g., "💜✈").
    5. Replace HTML escaped characters (e.g., "&lt;").
    6. Remove punctuation (e.g., "!!!", "?!").
    7. Normalize times and dates (e.g., "2/24", "7:00 AM").
    8. Remove URLs (e.g., "http://t.co/...").
    9. Perform verb lemmatization.
    10. Normalize repeated characters in words (e.g., "soooo" -> "soo").
    11. Remove extra whitespace.
    12. Convert to lowercase.
    """
    # 1. Remove mentions
    text = re.sub(r"@\w+", "", text)

    # 2. Remove currency symbols and amounts
    text = re.sub(r"\$\d+(?:\.\d+)?", "", text)

    # 3. Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

    # 4. Remove emojis (basic Unicode range for emojis)
    text = re.sub(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", "", text)

    # 5. Replace HTML escaped characters
    text = re.sub(r"&\w+;", " ", text)

    # 6. Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # 7. Normalize times and dates
    text = re.sub(r"(\b\d{1,2}:\d{2}(?:[APap][Mm])?\b)|(\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b)", "", text)

    # 8. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 9. Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    text = " ".join([lemmatizer.lemmatize(word, wordnet.VERB) for word in words])

    # 10. Normalize repeated characters in words
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 11. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 12. Convert to lowercase
    text = text.lower()

    return text


def part_2(tweets_df):
    """
    Cleans the 'text' column in the dataset, performs deduplication, and handles empty tweets.
    """

    # Apply the cleaning function to the 'text' column
    tweets_df['cleaned_text'] = tweets_df['text'].apply(clean_text)

    # Remove duplicate rows (where both cleaned_text and sentiment are equal)
    tweets_df = tweets_df.drop_duplicates(subset=['cleaned_text', 'airline_sentiment'], keep='first')

    # Remove rows with empty cleaned_text
    tweets_df = tweets_df[tweets_df['cleaned_text'].str.strip() != ""]

    # Return the cleaned DataFrame
    return tweets_df
