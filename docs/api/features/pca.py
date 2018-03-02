import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features.pca import PCADecomposition


def pca(X, y, outpath, **kwargs):
    # Create a new figure and axes
    _, ax = plt.subplots()

    viz = PCADecomposition(ax=ax, **kwargs)
    viz.fit_transform(X, y)
    viz.poof(outpath=outpath)


if __name__ == '__main__':

    # Load the credit data set
    data = pd.read_csv("../../../examples/data/credit/credit.csv")

    # Specify the features of interest
    features = [
            'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
            'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
            'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
            'jul_pay', 'aug_pay', 'sep_pay',
        ]

    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.default.as_matrix()

    # Instantiate the visualizer
    pca(X, y, "images/pca_projection_2d.png", scale=True, center=False, col=y)

    pca(X, y, "images/pca_projection_3d.png", scale=True, center=False, col=y, proj_dim=3)
