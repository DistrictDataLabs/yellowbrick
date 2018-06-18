import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from yellowbrick.regressor import ResidualsPlot


def plot_residuals(X, y, model, outpath="images/residuals.png", **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    _, ax = plt.subplots()

    visualizer = ResidualsPlot(model, ax=ax, **kwargs)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.poof(outpath=outpath)


def load_concrete():
    # Load the regression data set
    df = pd.read_csv("../../../examples/data/concrete/concrete.csv")

    feature_names = [
        'cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age'
    ]
    target_name = 'strength'

    # Get the X and y data from the DataFrame
    X = df[feature_names]
    y = df[target_name]

    return X, y



if __name__ == '__main__':
    # Draw the default residuals graph
    X, y = load_concrete()
    plot_residuals(X, y, Ridge())

    # Draw the residuals graph with no histogram
    plot_residuals(X, y, Ridge(), "images/residuals_no_hist.png", hist=False)
