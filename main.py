from part_1_a import part_1_a
from part_1_b import part_1_b
import pandas as pd

def part_1():
    tweets = 'Tweets.csv'
    relevant_columns = ['airline_sentiment', 'negativereason', 'airline', 'text']
    tweets_df = pd.read_csv(tweets, usecols=relevant_columns)
    part_1_a(tweets_df)
    part_1_b(tweets_df)

def main():
    part_1()

if __name__ == "__main__":
    main()