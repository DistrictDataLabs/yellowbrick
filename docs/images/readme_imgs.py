
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.base import VisualizerGrid
from yellowbrick.regressor import AlphaSelection
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import DiscriminationThreshold

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, Ridge


FIXTURES = "../../yellowbrick/datasets/fixtures"
SPAM_DATASET = os.path.join(FIXTURES, "spam", "spam.csv")
GAME_DATASET = os.path.join(FIXTURES, "game", "game.csv")
CONCRETE_DATASET = os.path.join(FIXTURES, "concrete", "concrete.csv")


def load_spam():
    spam = pd.read_csv(SPAM_DATASET)

    target = 'is_spam'
    features = [col for col in spam.columns if col != target]

    X = spam[features]
    y = spam[target]

    return X, y

def load_game():
    game = pd.read_csv(GAME_DATASET)

    classes = ["win", "loss", "draw"]
    game.replace({'loss':-1, 'draw':0, 'win':1, 'x':2, 'o':3, 'b':4}, inplace=True)

    X = game.iloc[:, game.columns != 'outcome']
    y = game['outcome']

    return X, y, classes

def load_concrete():
    concrete = pd.read_csv(CONCRETE_DATASET)

    feature_names = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age']
    target_name = 'strength'

    # Get the X and y data from the DataFrame
    X = concrete[feature_names]
    y = concrete[target_name]

    return X, y


if __name__ == "__main__":

    # Plot the Classifiers
    fig, (axa, axb, axc) = plt.subplots(
        nrows=1, ncols=3, figsize=(20, 4)
    )

    X, y, classes = load_game()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    class_pred = ClassPredictionError(
        GaussianNB(), classes=classes, ax=axa
    )
    class_pred.fit(X_train, y_train)
    class_pred.score(X_test, y_test)
    class_pred.finalize()

    X, y = load_spam()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    conf_mtx = PrecisionRecallCurve(
        LogisticRegression(), ax=axb
    )
    conf_mtx.fit(X_train, y_train)
    conf_mtx.score(X_test, y_test)
    conf_mtx.finalize()

    disc_thres = DiscriminationThreshold(
        RandomForestClassifier(), ax=axc
    )
    disc_thres.fit(X_train, y_train)
    disc_thres.finalize()

    plt.tight_layout(pad=1.5)
    plt.savefig("readme_classifiers.png")


    # Plot the Regressors
    fig, (axd, axe, axf) = plt.subplots(
        nrows=1, ncols=3, figsize=(20, 4)
    )

    X, y = load_concrete()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    resids = ResidualsPlot(Ridge(), ax=axd)
    resids.fit(X_train, y_train)
    resids.score(X_test, y_test)
    resids.finalize()

    pe_error = PredictionError(Lasso(), ax=axe)
    pe_error.fit(X_train, y_train)
    pe_error.score(X_test, y_test)
    pe_error.finalize()

    alphas = np.logspace(-10, 1, 400)
    alpha_sel = AlphaSelection(LassoCV(alphas=alphas), ax=axf)
    alpha_sel.fit(X_train, y_train)
    alpha_sel.finalize()

    plt.tight_layout(pad=1.5)
    plt.savefig("readme_regressors.png")

    fig, (axg, axh, axi) = plt.subplots(
        nrows=1, ncols=3, figsize=(20, 4)
    )

    # Plot the Clusterers
    X, y = make_blobs(centers=12, n_samples=1000, n_features=16, shuffle=True)

    icdm = InterclusterDistance(KMeans(9), ax=axg)
    icdm.fit(X)
    icdm.finalize()

    elbow = KElbowVisualizer(KMeans(), k=(4,12), ax=axh)
    elbow.fit(X)
    elbow.finalize()

    sil_score = SilhouetteVisualizer(KMeans(9), ax=axi)
    sil_score.fit(X)
    sil_score.finalize()

    plt.tight_layout(pad=1.5)
    plt.savefig("readme_clusterers.png")