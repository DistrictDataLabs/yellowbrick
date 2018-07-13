import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features.pca import PCADecomposition


def pca(X, y, outpath, **kwargs):
    # Create a new figure and axes
    _, ax = plt.subplots()

    viz = PCADecomposition(ax=ax, **kwargs)
    viz.fit_transform(X, y)
    viz.poof(outpath=outpath)


def load_credit():
    # Load the credit data set
    data = pd.read_csv("../../../examples/data/credit/credit.csv")

    # Specify the features of interest
    target = "default"
    features = [col for col in data.columns if col != target]

    # Extract the numpy arrays from the data frame
    X = data[features]
    y = data[target]
    return X, y


def load_concrete():
    # Load the credit data set
    data = pd.read_csv("../../../examples/data/concrete/concrete.csv")

    # Specify the features of interest
    feature_names = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age']
    target_name = 'strength'

    # Get the X and y data from the DataFrame
    X = data[feature_names]
    y = data[target_name]

    return X, y


if __name__ == '__main__':

    # Draw PCA with credit data set
    X, y = load_credit()
    colors = np.array(['r' if yi else 'b' for yi in y])
    pca(X, y, "images/pca_projection_2d.png", scale=True, color=colors)
    pca(X, y, "images/pca_projection_3d.png", scale=True, color=colors, proj_dim=3)

    # Draw biplots with concrete data set
    X, y = load_concrete()
    pca(X, y, "images/pca_biplot_2d.png", scale=True, proj_features=True)
    pca(X, y, "images/pca_biplot_3d.png", scale=True, proj_features=True, proj_dim=3)
