#!/usr/bin/env python

"""
Generate images for the elbow plot documentation.
"""

# Import necessary modules
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer


def draw_elbow(path="images/elbow.png"):
    # Generate synthetic dataset with 8 blobs
    X, y = make_blobs(
        centers=8, n_features=12, n_samples=1000,
        shuffle=True, random_state=42
    )

    # Create a new figure to draw the clustering visualizer on
    _, ax = plt.subplots()

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, ax=ax, k=(4,12))

    visualizer.fit(X)                # Fit the data to the visualizer
    visualizer.poof(outpath=path)    # Draw/show/poof the data


def draw_calinski_harabaz(path="images/calinski_harabaz.png"):
    # Generate synthetic dataset with 8 blobs
    X, y = make_blobs(
        centers=8, n_features=12, n_samples=1000,
        shuffle=True, random_state=42
    )

    # Create a new figure to draw the clustering visualizer on
    _, ax = plt.subplots()

    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(
        model, ax=ax, k=(4,12),
        metric='calinski_harabaz', timings=False
    )
    visualizer.fit(X)                # Fit the data to the visualizer
    visualizer.poof(outpath=path)    # Draw/show/poof the data


if __name__ == '__main__':
    draw_elbow()
    draw_calinski_harabaz()
