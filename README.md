# Airline Sentiment Analysis and Preprocessing

This project performs sentiment analysis on the Twitter US Airline Sentiment dataset, combining exploratory data analysis, custom preprocessing, and machine learning for sentiment classification. Leveraging custom tokenization and cleaning pipelines, the classification is powered by a **Support Vector Machine (SVM)** model. Additional analyses uncover insights into user behaviors and geographical trends.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Pipeline Details](#pipeline-details)
  - [Data Analysis](#data-analysis)
  - [Data Cleaning](#data-cleaning)
  - [Model Training](#model-training)
  - [Additional Analyses](#additional-analyses)
- [Evaluation](#evaluation)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Overview

The Twitter US Airline Sentiment dataset contains labeled tweets expressing sentiments (positive, neutral, negative) about six airlines. This project emphasizes:

- **Data Analysis**: Examining sentiment trends, tweet lengths, and frequent sentiments.
- **Preprocessing**: Cleaning and normalizing text using a custom pipeline.
- **Model Training**: Classifying tweets with an SVM model while evaluating the impact of preprocessing.
- **Additional Analyses**: Investigating user behaviors, geographic patterns, and top keywords.

---

## Requirements

To run this project, the following dependencies are required:

- Python >= 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Regex

### Installation of Dependencies

Install the necessary packages via:

```bash
pip install -r requirements.txt
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/airline-sentiment-analysis.git
cd airline-sentiment-analysis
```

### Step 2: Set Up a Virtual Environment

To avoid package conflicts, create a virtual environment:

```bash
python3 -m venv venv
# Activate on MacOS/Linux
source venv/bin/activate
# Activate on Windows
venv\Scripts\activate
```

---

## Dataset

### File Structure

Ensure the dataset file `Tweets.csv` is located in the project directory.

**Example Data Format:**

| airline_sentiment | airline | text                                      | name     | tweet_location | airline_sentiment_confidence |
| ----------------- | ------- | ----------------------------------------- | -------- | -------------- | ---------------------------- |
| negative          | Delta   | "Flight delayed for 2 hours. Unacceptable." | john_doe | Philadelphia   | 0.98                         |

---

## Usage

### Running the Complete Pipeline

To execute the full analysis, cleaning, model training, and additional tasks:

```bash
python main.py
```

---

## Pipeline Details

### Data Analysis

1. Compute key statistics for each airline:
   - Total tweets.
   - Most frequent sentiments and their counts.
   - Shortest and longest tweet lengths.
2. Generate:
   - Tweet length histograms (binned by 5 characters).
   - Sentiment distribution grid for all airlines.

### Data Cleaning

The preprocessing pipeline includes:

- Removal: Mentions, URLs, emojis, HTML characters, punctuation, duplicate rows, and empty tweets.
- Normalization: Dates, times, repeated characters, and text casing.
- Lemmatization: Standardize verbs to their base forms.

### Model Training

1. Encode cleaned tweets using TF-IDF vectors.
2. Train an SVM classifier with:
   - Loss: Hinge
   - Penalty: L2
   - Learning Rate: 0.001
3. Perform:
   - 10-fold cross-validation.
   - Ablation studies to assess preprocessing impacts.
4. Outputs:
   - Classification accuracy, precision, recall, and F1 scores.
   - Confusion matrix.

### Additional Analyses

1. Identify top-5 words for each user via TF-IDF, saved as `top_words_per_user.csv`.
2. Determine the most active users for each airline and analyze their tweets and sentiment.
3. Parse geographical data to identify tweets from variations of “Philadelphia.”
4. Create a subset of tweets with confidence > 0.6, saved as `high_confidence_subset.csv`.

---

## Evaluation

### Metrics

Model performance is evaluated using:

- Classification Accuracy
- Precision, Recall, F1 Scores (class-level and overall)
- Confusion Matrix

**Example Results:**

| Metric               | Value  |
| -------------------- | ------ |
| Accuracy             | 78.88% |
| Precision (Negative) | 81%    |
| Recall (Neutral)     | 46%    |
| F1-Score (Positive)  | 70%    |

---

## Limitations and Future Work

### Current Limitations

- **Dataset Imbalance**: The prevalence of negative sentiments impacts classifier performance.
- **Neutral Sentiment Challenges**: Neutral tweets are harder to classify due to overlaps with positive and negative sentiments.

### Future Work

- **Advanced Models**: Experiment with Transformer-based models like BERT for enhanced accuracy.
- **Hyperparameter Optimization**: Use grid or random search for better parameter tuning.
- **Adaptive Cleaning Pipelines**: Tailor preprocessing steps to tweet characteristics for improved results.

---

This project highlights the importance of preprocessing and analytical techniques in sentiment classification, emphasizing the balance between noise reduction and context preservation.

---
