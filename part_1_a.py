import pandas as pd
import matplotlib.pyplot as plt
import math

def part_1_a():
    tweets = 'Tweets.csv'
    relevant_columns = ['airline_sentiment', 'negativereason', 'airline', 'text']
    tweets_df = pd.read_csv(tweets, usecols=relevant_columns)

    for airline in tweets_df['airline'].unique():
        # Filter tweets for the current airline
        airline_data = tweets_df[tweets_df['airline'] == airline]
        
        # Count the total tweets for this airline
        count = airline_data.shape[0]
        print(f'Tweets for {airline}: {count}')
        
        # Determine unique values and most frequent value for 'airline_sentiment'
        sentiment_counts = airline_data['airline_sentiment'].value_counts()
        most_frequent_sentiment = sentiment_counts.idxmax()
        most_frequent_sentiment_count = sentiment_counts.max()
        print(f"Unique sentiments for {airline}: {sentiment_counts.index.tolist()}")
        print(f"Most frequent sentiment: {most_frequent_sentiment} ({most_frequent_sentiment_count} occurrences)")

        # Determine unique values and most frequent value for 'negativereason'
        if 'negativereason' in airline_data.columns:
            reason_counts = airline_data['negativereason'].value_counts()
            if not reason_counts.empty:
                most_frequent_reason = reason_counts.idxmax()
                most_frequent_reason_count = reason_counts.max()
                print(f"Unique negative reasons for {airline}: {reason_counts.index.tolist()}")
                print(f"Most frequent reason: {most_frequent_reason} ({most_frequent_reason_count} occurrences)")
            else:
                print(f"No negative reasons recorded for {airline}.")
        
        # Calculate tweet lengths
        tweet_lengths = airline_data['text'].apply(len)
        shortest_tweet = tweet_lengths.min()
        longest_tweet = tweet_lengths.max()
        print(f"Shortest tweet length for {airline}: {shortest_tweet}")
        print(f"Longest tweet length for {airline}: {longest_tweet}")

        # Plot tweet length distribution
        max_length = math.ceil(longest_tweet / 5) * 5  # Round up to nearest multiple of 5
        bins = list(range(0, max_length + 5, 5))  # Create bins of size 5
        
        plt.hist(tweet_lengths, bins=bins, edgecolor='black')
        plt.title(f"Tweet Length Distribution for {airline}")
        plt.xlabel("Tweet Length (in characters)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f"{airline}_tweet_length_histogram.png")  # Save the plot as an image
        plt.close()  # Close the plot to avoid overlap in the next iteration
        
        print(f"Histogram saved for {airline} as '{airline}_tweet_length_histogram.png'.\n\n")
