#!/usr/bin/env python
# manifold.py
# Produce images for manifold documentation.
#
# Author:  Benjamin Bengfort
# Created: Sat May 12 11:26:18 2018 -0400
#
# Copyright (C) 2018 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: manifold.py [] benjamin@bengfort.com $

"""
Produce images for manifold documentation.
"""

##########################################################################
## Imports
##########################################################################

import os
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

from yellowbrick.datasets import load_occupancy, load_concrete
from yellowbrick.features.manifold import Manifold, MANIFOLD_ALGORITHMS, plot_manifold_embedding


SKIP = (
    "ltsa",  # produces no result
    "hessian",  # errors because of matrix
    "mds",  # uses way too much memory
)

FIXTURES = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "data")
)


def dataset_example(dataset="occupancy", manifold="all", path="images/", **kwargs):
    if manifold == "all":
        if path is not None and not os.path.isdir(path):
            "please specify a directory to save examples to"

        for algorithm in MANIFOLD_ALGORITHMS:
            if algorithm in SKIP:
                continue

            print("generating {} {} manifold".format(dataset, algorithm))
            fpath = os.path.join(path, "{}_{}_manifold.png".format(dataset, algorithm))
            try:
                dataset_example(dataset, algorithm, fpath)
            except Exception as e:
                print(
                    "could not visualize {} manifold on {} data: {}".format(
                        algorithm, dataset, e
                    )
                )
                continue

        # Break here!
        return

    # Create single example
    _, ax = plt.subplots(figsize=(9, 6))
    oz = Manifold(ax=ax, manifold=manifold, **kwargs)

    if dataset == "occupancy":
        X, y = load_occupancy()
    elif dataset == "concrete":
        X, y = load_concrete()
    else:
        raise Exception("unknown dataset '{}'".format(dataset))

    oz.fit_transform(X, y)
    oz.show(outpath=path)


def select_features_example(
    algorithm="isomap",
    path="images/occupancy_select_k_best_isomap_manifold.png",
    **kwargs
):
    _, ax = plt.subplots(figsize=(9, 6))

    model = Pipeline(
        [
            ("selectk", SelectKBest(k=3, score_func=f_classif)),
            ("viz", Manifold(ax=ax, manifold=algorithm, **kwargs)),
        ]
    )

    X, y = load_occupancy()
    model.fit_transform(X, y)
    model.named_steps["viz"].show(outpath=path)


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

        if os.path.isdir(path):
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
        _, ax = plt.subplots(figsize=(9, 6))
        path = self._make_path(path, "s_curve_{}_manifold.png".format(algorithm))

        oz = Manifold(ax=ax, manifold=algorithm, colors="nipy_spectral")

        oz.fit_transform(self.X, self.y)
        oz.show(outpath=path)

    def plot_all_manifolds(self, path="images"):
        """
        Plot all s-curve examples
        """
        for algorithm in MANIFOLD_ALGORITHMS:
            self.plot_manifold_embedding(algorithm)

    def manifold_quick_method(image="manifold_quick_method.png"):

        X, y = load_concrete()

        _, ax = plt.subplots()

        # Instantiate the visualizer
        visualizer = manifold_embedding(X, y, manifold="isomap", n_neighbors=10)
        visualizer.show(outpath=os.path.join(IMAGES, image))

##########################################################################
## Main Method
##########################################################################

if __name__ == "__main__":
    # curve = SCurveExample()
    # curve.plot_all_manifolds()
    dataset_example("concrete", "tsne", path="images/concrete_tsne_manifold.png")
    dataset_example("occupancy", "tsne", path="images/occupancy_tsne_manifold.png")
    dataset_example(
        "concrete", "isomap", path="images/concrete_isomap_manifold.png", n_neighbors=10
    )
    select_features_example(algorithm="isomap", n_neighbors=10)
    manifold_quick_method()
