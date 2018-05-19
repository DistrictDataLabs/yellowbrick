import os
import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.classifier import DiscriminationThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from functools import partial

BASE = os.path.join("..", "..", "..", os.path.dirname("__file__"))
EXAMPLES = os.path.join(BASE, "examples", "data")

# TODO: Make these examples part of the code base
CHURN_DATASET = os.path.join(EXAMPLES, "churn", "churn.txt")
SPAM_DATASET =  os.path.join(EXAMPLES, "spam", "spam.csv")


def load_spam():
    df = pd.read_csv(SPAM_DATASET)

    target = 'is_spam'
    features = [col for col in df.columns if col != target]

    X = df[features]
    y = df[target]

    return X, y


def load_churn():
    df = pd.read_csv(CHURN_DATASET)
    df.columns = [
        c.lower().replace(' ', '_').replace('?', '').replace("'", "")
        for c in df.columns
    ]

    state_encoder = LabelEncoder()
    df.state = state_encoder.fit_transform(df.state)

    del df['phone']

    for col in ['intl_plan', 'vmail_plan', 'churn']:
        df[col] = df[col].map({'no': 0, 'False.': 0, 'yes': 1, 'True.': 1})

    X = df[[c for c in df.columns if c != 'churn']]
    y = df['churn']

    return X, y


def plot_discrimination_threshold(clf, data='spam', outpath=None):
    if data == 'spam':
        X, y = load_spam()
    elif data == 'churn':
        X, y = load_churn()
    else:
        raise ValueError("no dataset loader '{}'".format(data))

    _, ax = plt.subplots()

    visualizer = DiscriminationThreshold(clf, ax=ax)
    visualizer.fit(X, y)
    visualizer.poof(outpath=outpath)


plot_churn = partial(
    plot_discrimination_threshold, data='churn',
    outpath="images/churn_discrimination_threshold.png"
)

plot_spam = partial(
    plot_discrimination_threshold, data='spam',
    outpath="images/spam_discrimination_threshold.png"
)


if __name__ == '__main__':
    plot_churn(RandomForestClassifier())
    plot_spam(LogisticRegression())
