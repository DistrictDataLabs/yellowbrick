#!/usr/bin/env python

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from yellowbrick.classifier import PrecisionRecallCurve


# Location of downloaded datasets from Yellowbrick
FIXTURES = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "yellowbrick", "datasets", "fixtures"
)


def load_binary(split=True):
    data = pd.read_csv(os.path.join(FIXTURES, "spam", "spam.csv"))

    target = "is_spam"
    features = [col for col in data.columns if col != target]

    X = data[features]
    y = data[target]

    if split:
        return train_test_split(X, y, test_size=0.2, shuffle=True)
    return X, y


def load_multiclass(split=True):
    data = pd.read_csv(os.path.join(FIXTURES, "game", "game.csv"))

    # Encode the categorical variables
    data.replace({'x':0, 'o':1, 'b':2}, inplace=True)

    # Extract the numpy arrays from the data frame
    X = data.iloc[:, data.columns != 'outcome']
    y = LabelEncoder().fit_transform(data['outcome'])

    if split:
        return train_test_split(X, y, test_size=0.2, shuffle=True)
    return X, y


def draw_binary(outpath=None):
    _, ax = plt.subplots(figsize=(9,6))

    X_train, X_test, y_train, y_test = load_binary(split=True)

    oz = PrecisionRecallCurve(RidgeClassifier(), ax=ax)
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    oz.poof(outpath=outpath)


def draw_multiclass(outpath=None, simple=True):
    _, ax = plt.subplots(figsize=(9,6))

    X_train, X_test, y_train, y_test = load_multiclass()

    if simple:
        oz = PrecisionRecallCurve(RandomForestClassifier(), ax=ax)
    else:
        oz = PrecisionRecallCurve(MultinomialNB(), ax=ax, per_class=True, iso_f1_curves=True, fill_area=False, micro=False)

    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    oz.poof(outpath=outpath)



if __name__ == '__main__':
    draw_binary(outpath="images/binary_precision_recall.png")
    draw_multiclass(simple=True, outpath="images/multiclass_precision_recall.png")
    draw_multiclass(simple=False, outpath="images/multiclass_precision_recall_full.png")
