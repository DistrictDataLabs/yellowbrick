#!/usr/bin/env python
# manifold.py
# Produce images for manifold documentation.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sat May 12 11:26:18 2018 -0400
#
# ID: manifold.py [] benjamin@bengfort.com $

"""
Produce images for manifold documentation.
"""

##########################################################################
## Imports
##########################################################################

import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif#, mutual_info_classif
from yellowbrick.features.manifold import Manifold, MANIFOLD_ALGORITHMS

SKIP = (
    'ltsa', # produces no result
    'hessian', # errors because of matrix
    'mds', # uses way too much memory
)

FIXTURES = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "examples", "data"
))


def load_occupancy_data():
    # Load the classification data set
    data = pd.read_csv(os.path.join(FIXTURES, 'occupancy', 'occupancy.csv'))

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]

    X = data[features]
    y = pd.Series(['occupied' if y == 1 else 'unoccupied' for y in data.occupancy])

    return X, y


def load_concrete_data():
    # Load a regression data set
    data = pd.read_csv(os.path.join(FIXTURES, 'concrete', 'concrete.csv'))

    # Specify the features of interest
    feature_names = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age']
    target_name = 'strength'

    # Get the X and y data from the DataFrame
    X = data[feature_names]
    y = data[target_name]

    return X, y


def dataset_example(dataset="occupancy", manifold="all", path="images/"):
    if manifold == "all":
        if path is not None and not os.path.isdir(path):
            "please specify a directory to save examples to"

        for algorithm in MANIFOLD_ALGORITHMS:
            if algorithm in SKIP: continue

            print("generating {} {} manifold".format(dataset, algorithm))
            fpath = os.path.join(path, "{}_{}_manifold.png".format(dataset, algorithm))
            try:
                dataset_example(dataset, algorithm, fpath)
            except Exception as e:
                print("could not visualize {} manifold on {} data: {}".format(algorithm, dataset, e))
                continue


        # Break here!
        return

    # Create single example
    _, ax = plt.subplots(figsize=(9,6))
    oz = Manifold(ax=ax, manifold=manifold)

    if dataset == "occupancy":
        X, y = load_occupancy_data()
    elif dataset == "concrete":
        X, y = load_concrete_data()
    else:
        raise Exception("unknown dataset '{}'".format(dataset))

    oz.fit(X, y)
    oz.poof(outpath=path)


def select_features_example(algorithm='isomap', path="images/occupancy_select_k_best_isomap_manifold.png"):
    _, ax = plt.subplots(figsize=(9,6))

    model = Pipeline([
        ("selectk", SelectKBest(k=3, score_func=f_classif)),
        ("viz", Manifold(ax=ax, manifold=algorithm)),
    ])

    X, y = load_occupancy_data()
    model.fit(X, y)
    model.named_steps['viz'].poof(outpath=path)


class SCurveExample(object):
    """
    Creates an S-curve example and multiple visualizations
    """

    def __init__(self, n_points=1000, random_state=42):
        self.X, self.y = datasets.samples_generator.make_s_curve(
            n_points, random_state=random_state
        )

    def _make_path(self, path, name):
        """
        Makes directories as needed
        """
        if not os.path.exists(path):
            os.mkdirs(path)

        if os.path.isdir(path) :
            return os.path.join(path, name)

        return path

    def plot_original_3d(self, path="images"):
        """
        Plot the original data in 3-dimensional space
        """
        raise NotImplementedError("nyi")

    def plot_manifold_embedding(self, algorithm="lle", path="images"):
        """
        Draw the manifold embedding for the specified algorithm
        """
        _, ax = plt.subplots(figsize=(9,6))
        path = self._make_path(path, "s_curve_{}_manifold.png".format(algorithm))

        oz = Manifold(
            ax=ax, manifold=algorithm,
            target='continuous', colors='nipy_spectral'
        )

        oz.fit(self.X, self.y)
        oz.poof(outpath=path)

    def plot_all_manifolds(self, path="images"):
        """
        Plot all s-curve examples
        """
        for algorithm in MANIFOLD_ALGORITHMS:
            self.plot_manifold_embedding(algorithm)


if __name__ == '__main__':
    # curve = SCurveExample()
    # curve.plot_all_manifolds()

    dataset_example('occupancy', 'tsne', path="images/occupancy_tsne_manifold.png")
    # dataset_example('concrete', 'all')

    # select_features_example()
