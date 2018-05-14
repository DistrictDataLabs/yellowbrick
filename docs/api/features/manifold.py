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
from yellowbrick.features.manifold import Manifold, MANIFOLD_ALGORITHMS


FIXTURES = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "examples", "data"
))


def load_credit_data():
    # Load the classification data set
    data = pd.read_csv(os.path.join(FIXTURES, 'credit', 'credit.csv'))

    # Specify the features of interest
    features = [
        'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
        'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
        'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
        'jul_pay', 'aug_pay', 'sep_pay',
    ]

    # Extract the numpy arrays from the data frame
    X = data[features]
    y = data.default
    return X, y


def credit_example(manifold="all", path="images/"):
    if manifold == "all":
        if path is not None and not os.path.isdir(path):
            "please specify a directory to save example to"

        for algorithm in MANIFOLD_ALGORITHMS:
            fpath = os.path.join(path, "credit_{}_manifold.png".format(algorithm))
            try:
                credit_example(algorithm, fpath)
            except Exception as e:
                print("could not visualize {} manifold on credit data: {}".format(algorithm, e))
                continue


        # Break here!
        return

    # Create single example

    _, ax = plt.subplots(figsize=(9,6))
    oz = Manifold(ax=ax, manifold=manifold)
    oz.fit(*load_credit_data())
    oz.poof(outpath=path)



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
        oz = Manifold(ax=ax, manifold=algorithm)
        oz.fit(self.X, self.y)
        oz.poof(outpath=path)


if __name__ == '__main__':
    # curve = SCurveExample()
    # for algorithm in MANIFOLD_ALGORITHMS:
    #     curve.plot_manifold_embedding(algorithm)
    credit_example()
