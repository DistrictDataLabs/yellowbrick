import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features import ParallelCoordinates
from sklearn.datasets import load_iris


def load_occupancy_data():
    # Load the classification data set
    data = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

    # Specify the features of interest and the classes of the target
    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']

    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()

    return X, y, features, classes


def pcoords(X, y, outpath, **kwargs):
    # Create a new figure and axes
    _, ax = plt.subplots()

    # Create the visualizer
    visualizer = ParallelCoordinates(ax=ax, **kwargs)
    visualizer.fit_transform(X, y)

    # Save to disk
    visualizer.poof(outpath=outpath)


def plot_fast_vs_slow():
    data = load_iris()

    _, axes = plt.subplots(nrows=2, figsize=(9,9))

    for idx, fast in enumerate((False, True)):
        title = "Fast Parallel Coordinates" if fast else "Standard Parallel Coordinates"
        oz = ParallelCoordinates(ax=axes[idx], fast=fast, title=title)
        oz.fit_transform(data.data, data.target)
        oz.finalize()

    plt.tight_layout()
    plt.savefig("images/fast_vs_slow_parallel_coordinates.png")


def plot_speedup(trials=5, factors=np.arange(1, 11)):

    def pcoords_time(X, y, fast=True):
        _, ax = plt.subplots()
        oz = ParallelCoordinates(fast=fast, ax=ax)

        start = time.time()
        oz.fit_transform(X, y)
        delta = time.time() - start

        plt.cla()        # clear current axis
        plt.clf()        # clear current figure
        plt.close("all") # close all existing plots

        return delta

    def pcoords_speedup(X, y):
        fast_time = pcoords_time(X, y, fast=True)
        slow_time = pcoords_time(X, y, fast=False)

        return slow_time / fast_time

    data = load_iris()

    speedups = []
    variance = []

    for factor in factors:
        X = np.repeat(data.data, factor, axis=0)
        y = np.repeat(data.target, factor, axis=0)

        local_speedups = []
        for trial in range(trials):
            local_speedups.append(pcoords_speedup(X, y))

        local_speedups = np.array(local_speedups)
        speedups.append(local_speedups.mean())
        variance.append(local_speedups.std())

    speedups = np.array(speedups)
    variance = np.array(variance)

    series = pd.Series(speedups, index=factors)
    _, ax = plt.subplots(figsize=(9,6))
    series.plot(ax=ax, marker='o', label="speedup factor", color='b')

    # Plot one standard deviation above and below the mean
    ax.fill_between(
        factors, speedups - variance, speedups + variance, alpha=0.25,
        color='b',
    )

    ax.set_ylabel("speedup factor")
    ax.set_xlabel("dataset size (number of repeats in Iris dataset)")
    ax.set_title("Speed Improvement of Fast Parallel Coordinates")
    plt.savefig("images/fast_parallel_coordinates_speedup.png")


if __name__ == '__main__':
    # plot_fast_vs_slow()
    # plot_speedup()

    # Occupancy data visualizations
    X, y, features, classes = load_occupancy_data()

    # Draw the full, original parallel coordinates
    pcoords(
        X, y, "images/parallel_coordinates.png",
        classes=classes, features=features,
        sample=0.05, shuffle=True, random_state=19,
    )

    # Draw the noramlized, sampled parallel coordinates
    pcoords(
        X, y, "images/normalized_sampled_parallel_coordinates.png",
        classes=classes, features=features,
        normalize='standard', sample=0.05, shuffle=True, random_state=19,
    )
