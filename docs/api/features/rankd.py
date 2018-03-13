import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features.rankd import Rank1D, Rank2D

def rank1d(X, y, outpath, **kwargs):
    # Create a new figure and axes
    _, ax = plt.subplots()

    # Create the visualizer
    visualizer = Rank1D(ax=ax, **kwargs)
    visualizer.fit(X, y)
    visualizer.transform(X)

    # Save to disk
    visualizer.poof(outpath=outpath)


def rank2d(X, y, outpath, **kwargs):
    # Create a new figure and axes
    _, ax = plt.subplots()

    # Create the visualizer
    visualizer = Rank2D(ax=ax, **kwargs)
    visualizer.fit(X, y)
    visualizer.transform(X)

    # Save to disk
    visualizer.poof(outpath=outpath)


if __name__ == '__main__':
    # Load the regression data set
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

    # Instantiate the visualizer with the Shapiro-Wilk ranking algorithm
    rank1d(X, y, "images/rank1d_shapiro.png", features=features, algorithm='shapiro')

    # Instantiate the visualizer with the Covariance ranking algorithm
    rank2d(X, y, "images/rank2d_covariance.png", features=features, algorithm='covariance')

    # Instantiate the visualizer with the Pearson ranking algorithm
    rank2d(X, y, "images/rank2d_pearson.png", features=features, algorithm='pearson')
