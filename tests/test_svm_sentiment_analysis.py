"""Unit tests for the SVM sentiment analysis example."""

from examples.sentiment_analysis.svm_sentiment_analysis import (
    LABEL_NAMES,
    predict_sentiment,
    train_model,
)


def test_model_trains_and_predicts_valid_labels():
    """The model should train and return valid sentiment labels."""
    x_train = [
        "I love this product",
        "This is excellent",
        "I hate this product",
        "This is terrible",
        "The product is okay",
        "It is an average experience",
    ]

    y_train = [2, 2, 0, 0, 1, 1]

    vectorizer, model = train_model(x_train, y_train)

    x_test = [
        "This product is excellent",
        "This product is terrible",
        "This product is okay",
    ]

    predictions = predict_sentiment(vectorizer, model, x_test)

    assert len(predictions) == len(x_test)
    assert all(prediction in LABEL_NAMES for prediction in predictions)


def test_label_names_have_three_classes():
    """The example should define negative, neutral, and positive classes."""
    assert LABEL_NAMES == ["negative", "neutral", "positive"]
    assert len(LABEL_NAMES) == 3