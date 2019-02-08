#!/usr/bin/env python3
# Generates images for the gallery

import os
import argparse
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

from yellowbrick.datasets import load_occupancy, load_credit, load_concrete

from yellowbrick.features import RFECV, JointPlot
from yellowbrick.features import RadViz, Rank1D, Rank2D, ParallelCoordinates
from yellowbrick.features import PCADecomposition, Manifold, FeatureImportances

from yellowbrick.contrib.scatter import ScatterVisualizer

from sklearn.ensemble import RandomForestClassifier


GALLERY = path.join(path.dirname(__file__), "images", "gallery")


##########################################################################
## Helper Methods
##########################################################################

def newfig():
    """
    Helper function to create an axes object of the gallery dimensions.
    """
    # NOTE: this figsize generates a better thumbnail
    _, ax = plt.subplots(figsize=(8,4))
    return ax


def savefig(viz, name, gallery=GALLERY):
    """
    Saves the figure to the gallery directory
    """
    if not path.exists(gallery):
        os.makedirs(gallery)

    # Must save as png
    if len(name.split(".")) > 1:
        raise ValueError("name should not specify extension")

    outpath = path.join(gallery, name+".png")
    viz.poof(outpath=outpath)
    print("created {}".format(outpath))


##########################################################################
## Feature Analysis
##########################################################################

def radviz():
    X, y = load_occupancy()
    oz = RadViz(ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "radviz")


def rank1d():
    X, y = load_credit()
    oz = Rank1D(algorithm="shapiro", ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "rank1d_shapiro")


def rank2d():
    X, y = load_credit()
    oz = Rank2D(algorithm="covariance", ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "rank2d_covariance")


def pcoords():
    X, y = load_occupancy()
    oz = ParallelCoordinates(sample=0.05, shuffle=True, ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "parallel_coordinates")


def pca():
    X, y = load_credit()
    colors = np.array(['r' if yi else 'b' for yi in y])
    oz = PCADecomposition(scale=True, color=colors, proj_dim=3)
    oz.fit_transform(X, y)
    savefig(oz, "pca_projection_3d")


def manifold(dataset, manifold):
    if dataset == "concrete":
        X, y = load_concrete()
    elif dataset == "occupancy":
        X, y = load_occupancy()
    else:
        raise ValueError("unknown dataset")

    oz = Manifold(manifold=manifold, ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "{}_{}_manifold".format(dataset, manifold))


def importances():
    X, y = load_occupancy()
    oz = FeatureImportances(RandomForestClassifier(), ax=newfig())
    oz.fit(X, y)
    savefig(oz, "feature_importances")


def rfecv():
    X, y = load_credit()
    model = RandomForestClassifier(n_estimators=10)
    oz = RFECV(model, cv=3, scoring='f1_weighted', ax=newfig())
    oz.fit(X, y)
    savefig(oz, "rfecv_sklearn_example")


def scatter():
    X, y = load_occupancy()
    oz = ScatterVisualizer(x="light", y="CO2", ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "scatter")


def jointplot():
    X, y = load_concrete()
    oz = JointPlot(columns=["cement", "splast"], ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "jointplot")


if __name__ == "__main__":
    plots = {
        "all": None,
        "radviz": radviz,
        "rank1d": rank1d,
        "rank2d": rank2d,
        "pcoords": pcoords,
        "pca": pca,
        "concrete_tsne": lambda: manifold("concrete", "tsne"),
        "occupancy_tsne": lambda: manifold("occupancy", "tsne"),
        "concrete_isomap": lambda: manifold("concrete", "isomap"),
        "importances": importances,
        "rfecv": rfecv,
        "scatter": scatter,
        "jointplot": jointplot,
    }

    parser = argparse.ArgumentParser(description="gallery image generator")
    parser.add_argument(
        "plots", nargs="+", choices=plots.keys(), metavar="plot",
        help="names of images to generate"
    )
    args = parser.parse_args()

    queue = frozenset(args.plots)
    if "all" in queue:
        queue = frozenset(plots.keys())

    for item in queue:
        method = plots[item]
        if method is not None:
            method()