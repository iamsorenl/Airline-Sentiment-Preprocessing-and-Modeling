import pandas as pd
import matplotlib.pyplot as plt

def part_1_b(tweets_df):
    # Get unique airlines
    airlines = tweets_df['airline'].unique()
    sentiments = ['negative', 'neutral', 'positive']

    # Create a grid for the plots (2 rows x 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()  # Flatten the 2D array of axes for easy iteration

    # Colors for the sentiments
    colors = ['red', 'blue', 'green']

    # Plot for each airline
    for i, airline in enumerate(airlines):
        airline_data = tweets_df[tweets_df['airline'] == airline]
        sentiment_counts = airline_data['airline_sentiment'].value_counts()

        # Plot histogram
        axes[i].bar(sentiments, [sentiment_counts.get(s, 0) for s in sentiments], color=colors, edgecolor='black')
        axes[i].set_title(f"Sentiment Distribution for {airline}")
        axes[i].set_xlabel("Sentiments")
        axes[i].set_ylabel("Frequency")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust remaining empty subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Remove unused axes

    # Add a legend for sentiment colors
    fig.legend(sentiments, loc='upper center', ncol=3, title="Sentiments", bbox_to_anchor=(0.5, 1.05))

    # Save and show the plot
    plt.savefig("Sentiment_Distribution_Grid.png")
    plt.show()