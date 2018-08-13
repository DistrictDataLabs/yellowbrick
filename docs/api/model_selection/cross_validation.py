#!/usr/bin/env python3
# cross_validation.py

"""
Generates a CVScores image
"""

##########################################################################
## Imports
##########################################################################

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, StratifiedKFold

from yellowbrick.model_selection import CVScores


##########################################################################
## Helper Methods
##########################################################################

def load_occupancy():
    # Load the classification data set
    room = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

    features = ["temperature", "relative humidity", "light", "C02", "humidity"]

    # Extract the numpy arrays from the data frame
    X = room[features].values
    y = room.occupancy.values

    return X, y


def load_energy():
    # Load regression dataset
    energy = pd.read_csv('../../../examples/data/energy/energy.csv')

    targets = ["heating load", "cooling load"]
    features = [col for col in energy.columns if col not in targets]

    X = energy[features]
    y = energy[targets[1]]

    return X, y

def classification_cvscores(outpath="images/cv_scores_classifier.png", **kwargs):
    X, y = load_occupancy()

    # Create a new figure and axes
    _, ax = plt.subplots()

    cv = StratifiedKFold(12)

    oz = CVScores(
        MultinomialNB(), ax=ax, cv=cv, scoring='f1_weighted'
    )

    oz.fit(X, y)

    # Save to disk
    oz.poof(outpath=outpath)


def regression_cvscores(outpath="images/cv_scores_regressor.png", **kwargs):
    X, y = load_energy()

    # Create a new figure and axes
    _, ax = plt.subplots()

    cv = KFold(12)

    oz = CVScores(
        Ridge(), ax=ax, cv=cv, scoring='r2'
    )

    oz.fit(X, y)

    # Save to disk
    oz.poof(outpath=outpath)


if __name__ == '__main__':
    classification_cvscores()
    regression_cvscores()
