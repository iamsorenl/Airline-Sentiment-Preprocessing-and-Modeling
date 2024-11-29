import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet') # Download the WordNet corpus

def clean_text(text, stats, changes):
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
    original_text = text
    
    # 1. Remove mentions
    cleaned_text = re.sub(r"@\w+", "", text)
    if cleaned_text != text:
        stats['Remove Mentions'] += 1
        changes['Remove Mentions'].append((original_text, cleaned_text))
    text = cleaned_text

    # 2. Remove currency symbols and amounts
    cleaned_text = re.sub(r"\$\d+(?:\.\d+)?", "", text)
    if cleaned_text != text:
        stats['Remove Currency'] += 1
        changes['Remove Currency'].append((original_text, cleaned_text))
    text = cleaned_text

    # 3. Remove email addresses
    cleaned_text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)
    if cleaned_text != text:
        stats['Remove Emails'] += 1
        changes['Remove Emails'].append((original_text, cleaned_text))
    text = cleaned_text

    # 4. Remove emojis
    cleaned_text = re.sub(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]", "", text)
    if cleaned_text != text:
        stats['Remove Emojis'] += 1
        changes['Remove Emojis'].append((original_text, cleaned_text))
    text = cleaned_text

    # 5. Replace HTML escaped characters
    cleaned_text = re.sub(r"&\w+;", " ", text)
    if cleaned_text != text:
        stats['Replace HTML'] += 1
        changes['Replace HTML'].append((original_text, cleaned_text))
    text = cleaned_text

    # 6. Remove punctuation
    cleaned_text = re.sub(r"[^\w\s]", " ", text)
    if cleaned_text != text:
        stats['Remove Punctuation'] += 1
        changes['Remove Punctuation'].append((original_text, cleaned_text))
    text = cleaned_text

    # 7. Normalize times and dates
    cleaned_text = re.sub(r"(\b\d{1,2}:\d{2}(?:[APap][Mm])?\b)|(\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b)", "", text)
    if cleaned_text != text:
        stats['Normalize Dates/Times'] += 1
        changes['Normalize Dates/Times'].append((original_text, cleaned_text))
    text = cleaned_text

    # 8. Remove URLs
    cleaned_text = re.sub(r"http\S+|www\S+", "", text)
    if cleaned_text != text:
        stats['Remove URLs'] += 1
        changes['Remove URLs'].append((original_text, cleaned_text))
    text = cleaned_text

    # 9. Perform verb lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_text = " ".join([lemmatizer.lemmatize(word, wordnet.VERB) for word in words])
    if lemmatized_text != text:
        stats['Lemmatization'] += 1
        changes['Lemmatization'].append((original_text, lemmatized_text))
    text = lemmatized_text

    # 10. Normalize repeated characters
    cleaned_text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    if cleaned_text != text:
        stats['Normalize Repeated Characters'] += 1
        changes['Normalize Repeated Characters'].append((original_text, cleaned_text))
    text = cleaned_text

    # 11. Remove extra whitespace
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    if cleaned_text != text:
        stats['Remove Extra Whitespace'] += 1
        changes['Remove Extra Whitespace'].append((original_text, cleaned_text))
    text = cleaned_text

    # 12. Convert to lowercase
    cleaned_text = text.lower()
    if cleaned_text != text:
        stats['Convert to Lowercase'] += 1
        changes['Convert to Lowercase'].append((original_text, cleaned_text))
    text = cleaned_text

    return text


def part_2(tweets_df):
    """
    Cleans the 'text' column in the dataset, performs deduplication, and handles empty tweets.
    """

    # Initialize dictionaries to store statistics and changes made to the text
    stats = {step: 0 for step in [
        'Remove Mentions', 'Remove Currency', 'Remove Emails', 'Remove Emojis', 
        'Replace HTML', 'Remove Punctuation', 'Normalize Dates/Times', 
        'Remove URLs', 'Lemmatization', 'Normalize Repeated Characters', 
        'Remove Extra Whitespace', 'Convert to Lowercase'
    ]}
    changes = {step: [] for step in stats.keys()}

    # Apply the cleaning function to the 'text' column
    tweets_df['cleaned_text'] = tweets_df['text'].apply(lambda x: clean_text(x, stats, changes))

    # Remove duplicate rows (where both cleaned_text and sentiment are equal)
    tweets_df = tweets_df.drop_duplicates(subset=['cleaned_text', 'airline_sentiment'], keep='first')

    # Remove rows with empty cleaned_text
    tweets_df = tweets_df[tweets_df['cleaned_text'].str.strip() != ""]

    # Return the cleaned DataFrame
    return tweets_df, stats, changes
