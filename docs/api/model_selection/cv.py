#!/usr/bin/env python3
# cv.py

"""
Generates a CVScores image
"""

##########################################################################
## Imports
##########################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import CVScores


FIXTURES = os.path.join("..", "..", "..", "examples", "data")


##########################################################################
## Helper Methods
##########################################################################

def cv_scores_classifier(path="images/cv_scores_classifier.png"):

    data = pd.read_csv(os.path.join(FIXTURES, "game", "game.csv"))

    target = "outcome"
    features = [col for col in data.columns if col != target]

    X = pd.get_dummies(data[features])
    y = data[target]

    _, ax = plt.subplots()
    cv = StratifiedKFold(12)

    oz = CVScores(
        MultinomialNB(), ax=ax, cv=cv, scoring='f1_weighted'
    )

    oz.fit(X, y)
    oz.poof(outpath=path)


def cv_scores_regressor(path="images/cv_scores_regressor.png"):

    data = pd.read_csv(os.path.join(FIXTURES, "energy", "energy.csv"))

    targets = ["heating load", "cooling load"]
    features = [col for col in data.columns if col not in targets]

    X = data[features]
    y = data[targets[1]]

    _, ax = plt.subplots()

    oz = CVScores(RidgeCV(), ax=ax, scoring='r2')
    oz.fit(X, y)
    oz.poof(outpath=path)


##########################################################################
## Main Method
##########################################################################

if __name__ == '__main__':
    cv_scores_classifier()
    cv_scores_regressor()
