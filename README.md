# Airline Sentiment Analysis and Preprocessing

This project implements sentiment analysis on the Twitter US Airline Sentiment dataset. It includes data analysis, preprocessing, and machine learning model training for sentiment classification. The analysis leverages custom tokenization and cleaning pipelines, while the classification task is powered by a **Support Vector Machine (SVM)** using Scikit-learn. Additional analysis explores user behaviors and geographical insights.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
  - [Data Analysis](#data-analysis)
  - [Data Cleaning](#data-cleaning)
  - [Model Training](#model-training)
  - [Additional Analysis](#additional-analysis)
- [Evaluation](#evaluation)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

The Twitter US Airline Sentiment dataset is a labeled collection of tweets expressing customer sentiments (positive, negative, neutral) about various airlines. This project focuses on:

- **Data Analysis**: Understanding trends in sentiment distribution, tweet length, and frequent sentiments.
- **Preprocessing**: Cleaning and normalizing text data with custom pipelines.
- **Model Training**: Using SVM to classify tweets based on sentiment, with a focus on ablation studies for preprocessing steps.
- **Additional Analysis**: Exploring user behaviors, geographic insights, and top-words using TF-IDF.

---

## Requirements

To set up the project, ensure you have the following dependencies:

- Python >= 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Regex

### Installing Dependencies

You can install all necessary packages by running:

```bash
pip install -r requirements.txt
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/airline-sentiment-analysis.git
cd airline-sentiment-analysis
```

### 2. Create a Virtual Environment

To avoid package conflicts, it is recommended to use a virtual environment:

```bash
python3 -m venv venv
# Activate on MacOS/Linux
source venv/bin/activate
# Activate on Windows
venv\Scripts\activate
```

---

## Dataset Structure

Ensure the following files are present in the project directory:

- `Tweets.csv`: Twitter US Airline Sentiment dataset.

### Example Data Format

| airline_sentiment | airline | text                                        | name     | tweet_location | airline_sentiment_confidence |
| ----------------- | ------- | ------------------------------------------- | -------- | -------------- | ---------------------------- |
| negative          | Delta   | "Flight delayed for 2 hours. Unacceptable." | john_doe | Philadelphia   | 0.98                         |

---

## Usage

### Running the Analysis and Pipeline

To execute the full pipeline, including data analysis, cleaning, model training, and additional analysis:

```bash
python main.py
```

This command will:

1. Perform data analysis (tweet length distributions, sentiment distributions).
2. Clean and preprocess the dataset.
3. Train an SVM model for sentiment classification.
4. Generate insights for user and geographic analysis.

---

## Pipeline Stages

### Data Analysis

1. Compute statistics for each airline, including:
   - Total tweets.
   - Most frequent sentiments and their counts.
   - Length of shortest and longest tweets.
2. Generate:
   - Histogram for tweet lengths, binned in intervals of 5.
   - Sentiment distribution grid plot for all airlines.

### Data Cleaning

The cleaning pipeline includes:

- Removing mentions, currency symbols, URLs, emails, emojis, punctuation, and HTML characters.
- Normalizing dates, times, and repeated characters.
- Lemmatizing verbs.
- Removing duplicate rows and empty tweets.

### Model Training

1. Encode tweets using TF-IDF.
2. Train an **SVM classifier** with the following parameters:
   - Penalty: L2
   - Loss: Hinge
   - Learning Rate: 0.001
3. Perform:
   - 10-fold cross-validation.
   - Ablation study to evaluate the impact of preprocessing actions.
4. Output:
   - Classification accuracy, precision, recall, F1 scores.
   - Confusion matrix.

### Additional Analysis

1. Identify top-5 words for each user using TF-IDF, saved to `top_words_per_user.csv`.
2. Determine the most active users for each airline and extract their tweets, locations, and sentiments.
3. Analyze geographic data, identifying variations of "Philadelphia" in tweet locations.
4. Create and save a subset of tweets with sentiment confidence > 0.6.

---

## Evaluation

### Metrics

The model is evaluated using:

- **Classification Accuracy**
- **Class-level Precision, Recall, F1 Score**
- **Confusion Matrix**

### Example Results

| Metric               | Value  |
| -------------------- | ------ |
| Accuracy             | 78.88% |
| Precision (Negative) | 81%    |
| Recall (Neutral)     | 46%    |
| F1-Score (Positive)  | 70%    |

---

## Limitations and Future Work

### Current Limitations

- **Dataset Imbalance**: The dataset has a high proportion of negative tweets, which skews classification performance.
- **Neutral Sentiment Challenges**: Neutral tweets are harder to classify, given their overlap with positive and negative sentiments.

### Future Work

- **Advanced Models**: Incorporate Transformer-based models like BERT for improved performance.
- **Automated Hyperparameter Tuning**: Implement grid or random search for parameter optimization.
- **Dynamic Cleaning Pipelines**: Adjust cleaning actions based on tweet characteristics for better preprocessing.

---

This project highlights the importance of preprocessing and modeling decisions in sentiment classification tasks.
