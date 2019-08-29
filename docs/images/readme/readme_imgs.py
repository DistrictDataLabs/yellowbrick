#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from yellowbrick.regressor import AlphaSelection
from yellowbrick.regressor import ResidualsPlot
from yellowbrick.regressor import PredictionError
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.datasets import load_spam, load_concrete, load_game


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, Ridge


FIGSIZE = (20, 4)

IMAGES = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
YB_LOGO_PATH = os.path.join(IMAGES, "yb-fc.png")


def tts_plot(viz, X, y, test_size=0.20, random_state=42, score=True, finalize=True):
    """
    Helper function to plot model visualizers with train_test_split
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    viz.fit(X_train, y_train)
    if score:
        viz.score(X_test, y_test)
    if finalize:
        viz.finalize()

    return viz


def class_prediction_error(ax=None):
    data = load_game(return_dataset=True)
    X, y = data.to_numpy()

    X = OneHotEncoder().fit_transform(X).toarray()

    viz = ClassPredictionError(GaussianNB(), ax=ax)
    return tts_plot(viz, X, y)


def confusion_matrix(ax=None):
    data = load_spam(return_dataset=True)
    X, y = data.to_pandas()

    viz = PrecisionRecallCurve(LogisticRegression(), ax=ax)
    return tts_plot(viz, X, y)


def discrimination_threshold(ax=None):
    data = load_spam(return_dataset=True)
    X, y = data.to_pandas()

    viz = DiscriminationThreshold(RandomForestClassifier(n_estimators=10), ax=ax)
    return tts_plot(viz, X, y, score=False)


def classification_visualizers(saveto=None):
    _, (axa, axb, axc) = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE)

    class_prediction_error(axa)
    confusion_matrix(axb)
    discrimination_threshold(axc)

    plt.tight_layout(pad=1.5)

    if saveto is not None:
        plt.savefig(saveto)
    else:
        plt.show()


def residuals_plot(ax=None):
    data = load_concrete(return_dataset=True)
    X, y = data.to_pandas()

    viz = ResidualsPlot(Ridge(), ax=ax)
    return tts_plot(viz, X, y)


def prediction_error(ax=None):
    data = load_concrete(return_dataset=True)
    X, y = data.to_pandas()

    viz = PredictionError(Lasso(), ax=ax)
    return tts_plot(viz, X, y)


def alpha_selection(ax=None):
    data = load_concrete(return_dataset=True)
    X, y = data.to_pandas()

    alphas = np.logspace(-10, 1, 400)
    viz = AlphaSelection(LassoCV(alphas=alphas), ax=ax)
    return tts_plot(viz, X, y)


def regression_visualizers(saveto=None):
    _, (axa, axb, axc) = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE)

    residuals_plot(axa)
    prediction_error(axb)
    alpha_selection(axc)

    plt.tight_layout(pad=1.5)

    if saveto is not None:
        plt.savefig(saveto)
    else:
        plt.show()


def intercluster_distance(ax=None):
    X, y = make_blobs(centers=12, n_samples=1000, n_features=16, shuffle=True)

    viz = InterclusterDistance(KMeans(9), ax=ax)
    viz.fit(X)
    viz.finalize()

    return viz


def k_elbow(ax=None):
    X, y = make_blobs(centers=12, n_samples=1000, n_features=16, shuffle=True)

    viz = KElbowVisualizer(KMeans(), k=(4, 12), ax=ax, locate_elbow=False)
    viz.fit(X)
    viz.finalize()

    return viz


def silhouette(ax=None):
    X, y = make_blobs(centers=12, n_samples=1000, n_features=16, shuffle=True)

    viz = SilhouetteVisualizer(KMeans(9), ax=ax)
    viz.fit(X)
    viz.finalize()

    return viz


def clustering_visualizers(saveto=None):
    _, (axa, axb, axc) = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE)

    intercluster_distance(axa)
    k_elbow(axb)
    silhouette(axc).ax.get_legend().remove()

    plt.tight_layout(pad=1.5)

    if saveto is not None:
        plt.savefig(saveto)
    else:
        plt.show()


def yb_logo(path=YB_LOGO_PATH, ax=None):
    """
    Reads the YB image logo from the specified path and writes it to the axes.
    """
    # Load image
    with open(path, "rb") as fobj:
        img = plt.imread(fobj, format="png")

    if ax is None:
        _, ax = plt.subplots()

    # Draw image
    ax.imshow(img, interpolation="nearest")

    # Remove spines, ticks, grid, and other marks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for pos in ["right", "top", "bottom", "left"]:
        ax.spines[pos].set_visible(False)

    return ax


def full_image(saveto=None, center_logo=False):
    _, axes = plt.subplots(nrows=3, ncols=3, figsize=(21, 14))

    # Top row: classifiers
    class_prediction_error(axes[0][0])
    confusion_matrix(axes[0][1])
    discrimination_threshold(axes[0][2])

    # Middle row: regressors
    residuals_plot(axes[1][0])
    alpha_selection(axes[1][2])

    if center_logo:
        yb_logo(ax=axes[1][1])
    else:
        prediction_error(axes[1][1])

    # Bottom row: clusterers
    intercluster_distance(axes[2][0])
    k_elbow(axes[2][1])
    silhouette(axes[2][2]).ax.get_legend().remove()

    plt.tight_layout(pad=1.5)

    if saveto is not None:
        plt.savefig(saveto)
    else:
        plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="generates images for the README.md banner"
    )

    parser.add_argument(
        "-c",
        "--classifiers",
        type=str,
        metavar="PATH",
        default="classifiers.png",
        help="path to save the classifiers banner image",
    )

    parser.add_argument(
        "-r",
        "--regressors",
        type=str,
        metavar="PATH",
        default="regressors.png",
        help="path to save the regressors banner image",
    )

    parser.add_argument(
        "-C",
        "--clusterers",
        type=str,
        metavar="PATH",
        default="clusterers.png",
        help="path to save the clusterers banner image",
    )

    parser.add_argument(
        "-b",
        "--banner",
        type=str,
        metavar="PATH",
        default="",
        help="make full banner image and save to disk",
    )

    parser.add_argument(
        "-y",
        "--yb",
        action="store_true",
        help="replace middle image of banner with logo",
    )

    args = parser.parse_args()

    if args.banner:
        full_image(args.banner, args.yb)
        sys.exit(0)

    if args.classifiers:
        classification_visualizers(args.classifiers)

    if args.regressors:
        regression_visualizers(args.regressors)

    if args.clusterers:
        clustering_visualizers(args.clusterers)
