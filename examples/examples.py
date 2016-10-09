#!/usr/bin/env python
# Ben's scratchpad for testing

## Imports
import os
import pandas as pd
import yellowbrick as yb
import matplotlib.pyplot as plt

from pandas.tools.plotting import radviz, parallel_coordinates
from yellowbrick.features import ParallelCoordinates, RadViz, Rank2D

## Module Constants - the path to the test data sets
FIXTURES = os.path.join(os.path.dirname(__file__), "examples", "data")

## Dataset loading mechanisms
datasets = {
    "credit": os.path.join(FIXTURES, "credit.xls"),
    "concrete": os.path.join(FIXTURES, "concrete.xls"),
    "occupancy": os.path.join(FIXTURES, 'occupancy', 'datatraining.txt'),
}

## Human readable column names
columns  = {
    "credit": [
        'id', 'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
        'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
        'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
        'jul_pay', 'aug_pay', 'sep_pay', 'default'
    ],
    "concrete": [
        'cement', 'slag', 'ash', 'water', 'splast',
        'coarse', 'fine', 'age', 'strength'
    ],
    "occupancy": [
        'date', 'temp', 'humid', 'light', 'co2', 'hratio', 'occupied'
    ],
}


def load_data(name):
    """
    Loads and wrangls the passed in dataset.
    """

    path = datasets[name]
    data = {
        'credit': lambda p: pd.read_excel(p, header=1),
        'concrete': lambda p: pd.read_excel(p),
        'occupancy': lambda p: pd.read_csv(p),
    }[name](path)

    data.columns = columns[name]
    return data


def test_parallel_coords(pandas=False, outpath=None):
    """
    Runs the parallel coordinates visualizer on the dataset.

    Parameters
    ----------
    pandas : bool
        Run the pandas version of the function
    outpath : path or None
        Save the figure to disk rather than show (if None)
    """
    data = load_data('occupancy')       # Load the data
    features = ['temp', 'humid', 'light', 'co2', 'hratio']
    classes = ['unoccupied', 'occupied']
    X = data[features].as_matrix()
    y = data.occupied.as_matrix()

    if pandas:
        parallel_coordinates(data[features + ['occupied']], 'occupied')
        if outpath:
            plt.savefig(outpath)
        else:
            plt.show()

    else:
        visualizer = ParallelCoordinates(   # Instantiate the visualizer
            classes=classes, features=features
        )
        visualizer.fit(X, y)                # Fit the data to the visualizer
        visualizer.transform(X)             # Transform the data
        visualizer.poof(outpath=outpath)    # Draw/show/poof the data


def test_radviz(pandas=False, outpath=None):
    """
    Runs the radviz visualizer on the dataset.

    Parameters
    ----------
    pandas : bool
        Run the pandas version of the function
    outpath : path or None
        Save the figure to disk rather than show (if None)
    """
    data = load_data('occupancy')       # Load the data
    features = ['temp', 'humid', 'light', 'co2', 'hratio']
    classes = ['unoccupied', 'occupied']
    X = data[features].as_matrix()
    y = data.occupied.as_matrix()

    if pandas:
        radviz(data[features + ['occupied']], 'occupied')
        if outpath:
            plt.savefig(outpath)
        else:
            plt.show()

    else:
        visualizer = RadViz(   # Instantiate the visualizer
            classes=classes, features=features
        )
        visualizer.fit(X, y)                # Fit the data to the visualizer
        visualizer.transform(X)             # Transform the data
        visualizer.poof(outpath=outpath)    # Draw/show/poof the data


def test_rank2d(seaborn=False, outpath=None):
    """
    Runs the radviz visualizer on the dataset.

    Parameters
    ----------
    pandas : bool
        Run the pandas version of the function
    outpath : path or None
        Save the figure to disk rather than show (if None)
    """
    data = load_data('occupancy')       # Load the data
    features = ['temp', 'humid', 'light', 'co2', 'hratio']
    classes = ['unoccupied', 'occupied']
    X = data[features].as_matrix()
    y = data.occupied.as_matrix()

    if seaborn:
        raise NotImplementedError("Not yet!")

    else:
        visualizer = Rank2D(features=features, algorithm='covariance')
        visualizer.fit(X, y)                # Fit the data to the visualizer
        visualizer.transform(X)             # Transform the data
        visualizer.poof(outpath=outpath)    # Draw/show/poof the data


if __name__ == '__main__':
    # test_parallel_coords(pandas=True)
    # test_radviz(pandas=False, outpath='/Users/benjamin/Desktop/yb_radviz.png')
    test_rank2d(outpath='/Users/benjamin/Desktop/yb_rank2d_covariance.png')
