from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from part_2 import part_2

# Splitting Data for Model Training
def split_data(clean_data):
    """
    Splits the cleaned data into training and testing sets.
    """
    X = clean_data['cleaned_text']
    y = clean_data['airline_sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=3, shuffle=True)
    
    # Print the distribution of the training and test set
    print("Training set distribution:\n", y_train.value_counts())
    print("\nTest set distribution:\n", y_test.value_counts())

    return X_train, X_test, y_train, y_test

# Training and Evaluating the Model
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Trains an SVM classifier using SGD and evaluates its performance.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=100, random_state=3, shuffle=True, tol=None)

    # Perform 10-fold Cross-Validation
    cross_val_scores = cross_val_score(model, X_train_tfidf, y_train, cv=10, scoring='accuracy')
    print("\n10-Fold Cross-Validation Scores:", cross_val_scores)
    print("\nMean Validation Accuracy: {:.4f}".format(cross_val_scores.mean()))

    # Train and Evaluate on the Test Set
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    # Classification Metrics
    print("\nTest Set Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n")

def part_3(tweets_df):
    """
    Main function for Part 3.
    """
    # Clean the data
    clean_data = part_2(tweets_df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(clean_data)

    # Train and evaluate the model
    train_and_evaluate_model(X_train, X_test, y_train, y_test)