# Three-Class SVM Sentiment Analysis

This example demonstrates sentiment classification using TF-IDF features and a linear Support Vector Machine.

The model predicts three classes:

- Negative
- Neutral
- Positive

## Installation

From the Yellowbrick repository root, run:

```bash
python -m pip install datasets scikit-learn yellowbrick matplotlib
```

## Running the Python example

```bash
python examples/sentiment_analysis/svm_sentiment_analysis.py
```

## Running the notebook

Open the following notebook:

```text
examples/sentiment_analysis/svm_sentiment_analysis.ipynb
```

The notebook loads the TweetEval sentiment dataset, converts text into TF-IDF features, trains a linear SVM, reports evaluation metrics, and generates Yellowbrick visualizations.

## Method

TF-IDF converts text into numerical features. The linear SVM learns boundaries between negative, neutral, and positive text.

The example uses unigrams and bigrams, balanced class weights, and a fixed random seed. These settings make the experiment reproducible and help reduce the effect of unequal class frequencies.

## Evaluation

The notebook reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Yellowbrick classification report

## Dataset

The example uses the public TweetEval sentiment dataset from Cardiff NLP. The sentiment task contains negative, neutral, and positive labels.

## Limitations

The model may have difficulty with sarcasm, slang, spelling errors, short messages, and text that requires outside context. It is intended for education and demonstration, not for making high-impact decisions about individuals.
