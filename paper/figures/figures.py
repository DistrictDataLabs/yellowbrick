#!/usr/bin/env python3
# Script to create visualizations for the JOSS paper

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features import Rank2D, RadViz
from yellowbrick.model_selection import LearningCurve
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.classifier import ClassificationReport, DiscriminationThreshold
from yellowbrick.regressor import ResidualsPlot, PredictionError, AlphaSelection

from collections import namedtuple
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LassoCV, RidgeCV, LogisticRegression


# Store figures alongside the script that generates them
FIGURES = os.path.dirname(__file__)

# Path to datasets downloaded from S3
DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "yellowbrick", "datasets", "fixtures"
)

# Quick reference dataset objects
Dataset = namedtuple("Dataset", "X,y")
Split = namedtuple("Split", "train,test")


def _make_dataset(X, y, split=False):
    if split:
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
        return Dataset(Split(X_train, X_test), Split(y_train, y_test))
    return Dataset(X, y)


def load_occupancy(split=False):
    """
    Create a dataset for the specified yb dataset
    """
    path = os.path.join(DATA, "occupancy", "occupancy.csv")
    data = pd.read_csv(path)

    X = data[["temperature", "relative humidity", "light", "C02", "humidity"]]
    y = data["occupancy"]
    return _make_dataset(X, y, split)


def load_concrete(split=False):
    path = os.path.join(DATA, "concrete", "concrete.csv")
    data = pd.read_csv(path)

    X = data[["cement", "slag", "ash", "water", "splast", "coarse", "fine", "age"]]
    y = data["strength"]
    return _make_dataset(X, y, split)


def load_spam(split=False):
    path = os.path.join(DATA, "spam", "spam.csv")
    data = pd.read_csv(path)

    target = "is_spam"
    features = [col for col in data.columns if col != target]

    X = data[features]
    y = data[target]
    return _make_dataset(X, y, split)


def feature_analysis(fname="feature_analysis.png"):
    """
    Create figures for feature analysis
    """

    # Create side-by-side axes grid
    _, axes = plt.subplots(ncols=2, figsize=(18, 6))

    # Draw RadViz on the left
    data = load_occupancy(split=False)
    oz = RadViz(ax=axes[0], classes=["unoccupied", "occupied"])
    oz.fit(data.X, data.y)
    oz.finalize()

    # Draw Rank2D on the right
    data = load_concrete(split=False)
    oz = Rank2D(ax=axes[1])
    oz.fit_transform(data.X, data.y)
    oz.finalize()

    # Save figure
    path = os.path.join(FIGURES, fname)
    plt.tight_layout()
    plt.savefig(path)


def regression(fname="regression.png"):
    """
    Create figures for regression models
    """
    _, axes = plt.subplots(ncols=2, figsize=(18, 6))
    alphas = np.logspace(-10, 1, 300)
    data = load_concrete(split=True)

    # Plot prediction error in the middle
    oz = PredictionError(LassoCV(alphas=alphas), ax=axes[0])
    oz.fit(data.X.train, data.y.train)
    oz.score(data.X.test, data.y.test)
    oz.finalize()

    # Plot residuals on the right
    oz = ResidualsPlot(RidgeCV(alphas=alphas), ax=axes[1])
    oz.fit(data.X.train, data.y.train)
    oz.score(data.X.test, data.y.test)
    oz.finalize()

    # Save figure
    path = os.path.join(FIGURES, fname)
    plt.tight_layout()
    plt.savefig(path)


def classification(fname="classification.png"):

    # Create side-by-side axes grid
    _, axes = plt.subplots(ncols=2, figsize=(18, 6))

    # Add ClassificationReport to the reft
    data = load_spam(split=True)
    oz = ClassificationReport(MultinomialNB(), classes=["ham", "spam"], ax=axes[0])
    oz.fit(data.X.train, data.y.train)
    oz.score(data.X.test, data.y.test)
    oz.finalize()

    # Add DiscriminationThreshold to the right
    data = load_spam(split=False)
    oz = DiscriminationThreshold(LogisticRegression(), ax=axes[1])
    oz.fit(data.X, data.y)
    oz.finalize()

    # Save figure
    path = os.path.join(FIGURES, fname)
    plt.tight_layout()
    plt.savefig(path)


def clustering(fname="clustering.png"):
    # Create side-by-side axes grid
    _, axes = plt.subplots(ncols=2, figsize=(18, 6))
    X, y = make_blobs(centers=7)

    # Add K-Elbow to the left
    oz = KElbowVisualizer(MiniBatchKMeans(), k=(3, 12), ax=axes[0])
    oz.fit(X, y)
    oz.finalize()

    # Add SilhouetteVisualizer to the right
    oz = SilhouetteVisualizer(Birch(n_clusters=5), ax=axes[1])
    oz.fit(X, y)
    oz.finalize()

    # Save figure
    path = os.path.join(FIGURES, fname)
    plt.tight_layout()
    plt.savefig(path)


def hyperparameter_tuning(fname="hyperparameter_tuning.png"):
    # Create side-by-side axes grid
    _, axes = plt.subplots(ncols=2, figsize=(18, 6))

    # Load the concrete dataset
    data = load_concrete(split=False)

    # Create a list of alphas to cross-validate against
    alphas = np.logspace(-10, 1, 400)

    # Add AlphaSelection to the left
    oz = AlphaSelection(LassoCV(alphas=alphas), ax=axes[0])
    oz.fit(data.X, data.y)
    oz.finalize()

    # Add LearningCurve to the right
    oz = LearningCurve(RandomForestRegressor(), scoring="r2", ax=axes[1])
    oz.fit(data.X, data.y)
    oz.finalize()

    # Save figure
    path = os.path.join(FIGURES, fname)
    plt.tight_layout()
    plt.savefig(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="generate visualizations for JOSS paper"
    )

    args = parser.parse_args()
    feature_analysis()
    regression()
    classification()
    clustering()
    hyperparameter_tuning()
