from part_1.part_1_a import part_1_a
from part_1.part_1_b import part_1_b
from part_1.part_1_c import part_1_c
from part_1.part_1_d import part_1_d
import pandas as pd

def part_1(tweets_df):
    part_1_a(tweets_df)
    part_1_b(tweets_df)
    part_1_c(tweets_df)
    part_1_d(tweets_df)

def part_2(tweets_df):
    pass

def main():
    tweets = 'Tweets.csv'
    relevant_columns = ['airline_sentiment', 'negativereason', 'airline', 'text']
    tweets_df = pd.read_csv(tweets, usecols=relevant_columns)
    part_1(tweets_df)
    part_2(tweets_df)

if __name__ == "__main__":
    main()
