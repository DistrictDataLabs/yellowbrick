import numpy as np
import pandas as pd

from sklearn.linear_model import LassoCV

from yellowbrick.regressor import AlphaSelection


if __name__ == '__main__':
    # Load the regression data set
    df = pd.read_csv("../../../examples/data/concrete/concrete.csv")

    feature_names = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age']
    target_name = 'strength'

    # Get the X and y data from the DataFrame
    X = df[feature_names].as_matrix()
    y = df[target_name].as_matrix()

    # Instantiate the linear model and visualizer
    alphas = np.logspace(-10, 1, 400)
    visualizer = AlphaSelection(LassoCV(alphas=alphas))

    visualizer.fit(X, y)
    g = visualizer.poof(outpath="images/alpha_selection.png")
