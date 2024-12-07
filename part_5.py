from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re

def part_5(tweets_df):
    # Find number of unique users
    if 'name' not in tweets_df.columns:
        print("[ERROR] Dataset must include a 'name' column for unique user identification.")
        return
    
    unique_users = tweets_df['name'].nunique()
    print(f"[INFO] Number of unique users: {unique_users}")
    
    # Compute top-5 words for each user's tweets using TF-IDF
    user_groups = tweets_df.groupby('name')['text'].apply(lambda x: ' '.join(x))
    tfidf = TfidfVectorizer(max_features=5)
    top_words = {}
    for user, tweets in user_groups.items():
        if not tweets.strip():  # Skip users with empty or whitespace-only tweets
            continue
        top_words[user] = tfidf.fit([tweets]).get_feature_names_out()

    # Save top words for each user to a CSV
    top_words_df = pd.DataFrame.from_dict(top_words, orient='index', columns=[f'Word{i+1}' for i in range(5)])
    top_words_df.to_csv('top_words_per_user.csv')
    print("[INFO] Top 5 words for each user saved to 'top_words_per_user.csv'.\n")

    # Find the most active users for each airline
    if 'airline' not in tweets_df.columns:
        print("[ERROR] Dataset must include an 'airline' column for identifying active users.")
        return
    
    most_active_users = tweets_df.groupby(['airline', 'name']).size().reset_index(name='tweet_count')
    most_active_users = most_active_users.sort_values(['airline', 'tweet_count'], ascending=[True, False])
    most_active_users = most_active_users.groupby('airline').head(1)
    print("[INFO] Most active users for each airline:")
    print(most_active_users)

    # Extract tweets, location, and sentiment for each active user
    active_user_details = tweets_df[tweets_df['name'].isin(most_active_users['name'])].copy()
    print("\n[INFO] Details of most active users:")
    print(active_user_details[['name', 'airline', 'text', 'tweet_location', 'airline_sentiment']])

    # Count and drop missing values in tweet_location and user_timezone
    missing_tweet_location = tweets_df['tweet_location'].isna().sum() if 'tweet_location' in tweets_df.columns else 0
    missing_user_timezone = tweets_df['user_timezone'].isna().sum() if 'user_timezone' in tweets_df.columns else 0
    print(f"[INFO] Missing values - tweet_location: {missing_tweet_location}, user_timezone: {missing_user_timezone}")
    
    tweets_df = tweets_df.dropna(subset=['tweet_location', 'user_timezone'], how='any').copy()
    print("[INFO] Rows with missing values in 'tweet_location' and 'user_timezone' dropped.")

    # Parse tweet_created field
    if 'tweet_created' in tweets_df.columns:
        tweets_df.loc[:, 'tweet_created'] = pd.to_datetime(tweets_df['tweet_created'], errors='coerce').copy()
        print("[INFO] Parsed 'tweet_created' field into datetime format:")
        print(tweets_df['tweet_created'].head())

    # Find tweets from Philadelphia
    if 'tweet_location' in tweets_df.columns:
        philly_variations = tweets_df['tweet_location'].dropna().unique()
        philly_variations = [loc for loc in philly_variations if re.search(r'philadelphia', loc, re.IGNORECASE)]
        print(f"[INFO] Philadelphia spellings found: {philly_variations}")
        philly_tweets = tweets_df[tweets_df['tweet_location'].str.contains(r'philadelphia', na=False, case=False)]
        print(f"[INFO] Total tweets from Philadelphia: {len(philly_tweets)}")

    # Create subset with high sentiment confidence
    if 'airline_sentiment_confidence' in tweets_df.columns:
        high_confidence_subset = tweets_df[tweets_df['airline_sentiment_confidence'] > 0.6].copy()
        high_confidence_subset.to_csv('high_confidence_subset.csv', index=False)
        print("[INFO] Subset with airline_sentiment_confidence > 0.6 saved to 'high_confidence_subset.csv'.")
        print(f"[INFO] Number of rows in high confidence subset: {len(high_confidence_subset)}")
