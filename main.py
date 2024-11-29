from part_1.part_1_a import part_1_a
from part_1.part_1_b import part_1_b
from part_1.part_1_c import part_1_c
from part_1.part_1_d import part_1_d
from part_2 import part_2
from part_3 import part_3
from part_5 import part_5
import pandas as pd

def part_1(tweets_df):
    part_1_a(tweets_df)
    part_1_b(tweets_df)
    part_1_c(tweets_df)
    part_1_d(tweets_df)

def main():
    tweets = 'Tweets.csv'
    relevant_columns = [
        'airline_sentiment',
        'negativereason',
        'airline',
        'text',
        'name',
        'tweet_location',
        'user_timezone',
        'tweet_created',
        'airline_sentiment_confidence'
    ]   
    tweets_df = pd.read_csv(tweets, usecols=relevant_columns)
    part_1(tweets_df)
    pt2 = part_2(tweets_df)
    print("\n----------- Part 2 -----------\n")
    print(pt2.head(10))
    print("\n----------- Part 3 -----------\n")
    part_3(tweets_df)
    # part 4 was the report
    print("\n----------- Part 4 was the report -----------\n")
    print("\n----------- Part 5 -----------\n")
    part_5(tweets_df)


if __name__ == "__main__":
    main()
