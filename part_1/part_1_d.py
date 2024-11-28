import nltk
from part_1.part_1_c import custom_tokenizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')  # Ensure NLTK's tokenizer is available

def part_1_d(tweets_df):
    """
    Compare custom tokenizer with NLTK tokenizer and document differences.
    """

    differences = []

    for idx, row in tweets_df.iterrows():
        text = row['text']
        custom_tokens = custom_tokenizer(text)
        nltk_tokens = word_tokenize(text)

        # Find differences between the token lists
        if custom_tokens != nltk_tokens:
            differences.append({
                'text': text,
                'custom_tokens': custom_tokens,
                'nltk_tokens': nltk_tokens
            })

        # Stop after collecting 5 examples
        if len(differences) >= 5:
            break

    # Write differences and analysis to a text file
    with open("tokenizer_comparison.txt", "w") as file:
        file.write("Comparison of Custom Tokenizer and NLTK Tokenizer:\n")
        file.write("--------------------------------------------------\n\n")

        for i, diff in enumerate(differences):
            file.write(f"Example {i + 1}:\n")
            file.write(f"Original Text: {diff['text']}\n")
            file.write(f"Custom Tokenizer Output: {diff['custom_tokens']}\n")
            file.write(f"NLTK Tokenizer Output: {diff['nltk_tokens']}\n")
            file.write("\n")

        # Write analysis paragraph
        file.write("Analysis:\n")
        file.write(
            "The custom tokenizer differs from NLTK's tokenizer in several ways. "
            "The custom tokenizer explicitly handles mentions, hashtags, and emojis, whereas NLTK's tokenizer "
            "focuses on splitting text into words and punctuation. As a result, NLTK's tokenizer may split hashtags "
            "or miss emojis, which the custom tokenizer captures separately. These differences make the custom tokenizer "
            "better suited for social media text processing.\n"
        )

    print("Differences and analysis written to 'tokenizer_comparison.txt'.")
