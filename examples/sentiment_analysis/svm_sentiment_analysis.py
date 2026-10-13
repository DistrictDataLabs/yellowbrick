"""Three-class sentiment analysis using TF-IDF and a linear SVM."""

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


LABEL_NAMES = ["negative", "neutral", "positive"]


def load_sentiment_data(sample_size=3000, random_state=42):
    """Load a reproducible sample from the TweetEval sentiment dataset."""
    dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    train_data = dataset["train"].shuffle(seed=random_state)
    test_data = dataset["test"].shuffle(seed=random_state)

    train_size = min(sample_size, len(train_data))
    test_size = min(sample_size, len(test_data))

    train_data = train_data.select(range(train_size))
    test_data = test_data.select(range(test_size))

    x_train = train_data["text"]
    y_train = train_data["label"]

    x_test = test_data["text"]
    y_test = test_data["label"]

    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train):
    """Convert text into TF-IDF features and train a linear SVM."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)

    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(x_train_tfidf, y_train)

    return vectorizer, model


def predict_sentiment(vectorizer, model, texts):
    """Predict sentiment labels for new text examples."""
    text_features = vectorizer.transform(texts)
    predictions = model.predict(text_features)

    return [LABEL_NAMES[prediction] for prediction in predictions]


def main():
    """Train the model and print sample predictions."""
    x_train, x_test, y_train, y_test = load_sentiment_data()

    vectorizer, model = train_model(x_train, y_train)

    predictions = predict_sentiment(vectorizer, model, x_test[:10])

    for text, prediction in zip(x_test[:10], predictions):
        print(f"{prediction}: {text}")


if __name__ == "__main__":
    main()