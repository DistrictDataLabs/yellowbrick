import pytest

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


TEXTS = [
    "I loved this excellent experience",
    "The product works perfectly",
    "The service was wonderful",
    "I hated this terrible experience",
    "The product stopped working",
    "The service was disappointing",
    "The appointment is scheduled tomorrow",
    "The product has several features",
    "The result was ordinary",
]

LABELS = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
    "neutral",
]


@pytest.fixture
def trained_model():
    """Train a small TF-IDF and SVM model for testing."""

    x_train, _, y_train, _ = train_test_split(
        TEXTS,
        LABELS,
        test_size=0.33,
        random_state=42,
        stratify=LABELS,
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LinearSVC(random_state=42)),
        ]
    )

    model.fit(x_train, y_train)

    return model


def test_model_can_predict(trained_model):
    """The model should return a valid sentiment label."""

    prediction = trained_model.predict(
        ["This product is excellent"]
    )

    assert len(prediction) == 1
    assert prediction[0] in {"positive", "negative", "neutral"}


def test_model_returns_correct_number_of_predictions(trained_model):
    """The model should return one prediction per input message."""

    messages = [
        "This is wonderful",
        "This is terrible",
        "The event is tomorrow",
    ]

    predictions = trained_model.predict(messages)

    assert len(predictions) == len(messages)


def test_model_contains_expected_pipeline_steps(trained_model):
    """The pipeline should contain TF-IDF and SVM components."""

    assert "tfidf" in trained_model.named_steps
    assert "classifier" in trained_model.named_steps
    assert isinstance(
        trained_model.named_steps["classifier"],
        LinearSVC,
    )