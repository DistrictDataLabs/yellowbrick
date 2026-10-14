"""
SVM Text Classification with Yellowbrick.

This example classifies short text messages into positive, negative,
and neutral sentiment categories using TF-IDF features and a linear SVM.
Yellowbrick is used to visualize the confusion matrix and classification
report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from yellowbrick.classifier import ClassificationReport, ConfusionMatrix


RANDOM_STATE = 42
CLASSES = ["negative", "neutral", "positive"]
OUTPUT_DIR = Path("outputs")


def create_dataset():
    """Create a small reproducible sentiment dataset."""

    data = {
        "text": [
            "I absolutely loved this experience",
            "The service was excellent and friendly",
            "This product works perfectly",
            "I am very happy with the results",
            "The movie was enjoyable and inspiring",
            "The support team solved my problem quickly",
            "This is a wonderful experience",
            "I would definitely recommend this product",
            "The quality is amazing",
            "Everything worked better than expected",

            "I hated this experience",
            "The service was rude and disappointing",
            "This product stopped working",
            "I am very unhappy with the results",
            "The movie was boring and frustrating",
            "The support team ignored my problem",
            "This was a terrible experience",
            "I would not recommend this product",
            "The quality is extremely poor",
            "Everything worked worse than expected",

            "The experience was acceptable",
            "The service was provided as scheduled",
            "The product has several features",
            "I received the expected results",
            "The movie started at seven",
            "The support team answered my question",
            "This was an ordinary experience",
            "I may consider this product",
            "The quality is average",
            "The result was neither good nor bad",
        ],
        "sentiment": (
            ["positive"] * 10
            + ["negative"] * 10
            + ["neutral"] * 10
        ),
    }

    return pd.DataFrame(data)


def build_model():
    """Build the TF-IDF vectorizer and linear SVM classifier."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )

    classifier = LinearSVC(random_state=RANDOM_STATE)

    return vectorizer, classifier


def split_data(dataframe):
    """Create stratified training and testing datasets."""

    return train_test_split(
        dataframe["text"],
        dataframe["sentiment"],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=dataframe["sentiment"],
    )


def evaluate_model(model, x_test, y_test):
    """Print accuracy and classification metrics."""

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASSES,
            target_names=CLASSES,
            zero_division=0,
        )
    )

    return predictions


def create_visualizations(model, x_train, y_train, x_test, y_test):
    """Create and save Yellowbrick evaluation visualizations."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    confusion_visualizer = ConfusionMatrix(
        model,
        classes=CLASSES,
        cmap="Blues",
        force_model=True,
    )
    confusion_visualizer.fit(x_train, y_train)
    confusion_visualizer.score(x_test, y_test)
    confusion_visualizer.show(
        outpath=str(OUTPUT_DIR / "confusion_matrix.png")
    )
    plt.close("all")

    report_visualizer = ClassificationReport(
        model,
        classes=CLASSES,
        support=True,
        cmap="Blues",
        force_model=True,
    )

    report_visualizer.fit(x_train, y_train)
    report_visualizer.score(x_test, y_test)
    report_visualizer.show(
        outpath=str(OUTPUT_DIR / "classification_report.png")
    )
    plt.close("all")

    print("Saved both visualization files.")

def predict_new_messages(vectorizer, model):
    """Predict sentiment for new text messages."""

    new_messages = [
        "The product is fantastic and works perfectly",
        "This was a disappointing and terrible experience",
        "The appointment is scheduled for tomorrow",
    ]

    new_features = vectorizer.transform(new_messages)
    predictions = model.predict(new_features)

    results = pd.DataFrame(
        {
            "message": new_messages,
            "predicted_sentiment": predictions,
        }
    )

    print("\nPredictions for new messages:")
    print(results.to_string(index=False))

    return results


def main():
    """Run the complete SVM text-classification workflow."""

    dataframe = create_dataset()

    print("Dataset shape:", dataframe.shape)
    print("\nClass distribution:")
    print(dataframe["sentiment"].value_counts())

    x_train, x_test, y_train, y_test = split_data(dataframe)

    vectorizer, model = build_model()

    x_train_features = vectorizer.fit_transform(x_train)
    x_test_features = vectorizer.transform(x_test)

    model.fit(x_train_features, y_train)

    print("\nModel training completed.")

    evaluate_model(model, x_test_features, y_test)

    create_visualizations(
        model,
        x_train_features,
        y_train,
        x_test_features,
        y_test,
    )

    predict_new_messages(vectorizer, model)


if __name__ == "__main__":
    main()