import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from yellowbrick.regressor import AlphaSelection
from yellowbrick.regressor import ManualAlphaSelection


if __name__ == '__main__':
    # Load the regression data set
    df = pd.read_csv("../../../examples/data/concrete/concrete.csv")

    feature_names = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age']
    target_name = 'strength'

    # Get the X and y data from the DataFrame
    X = df[feature_names].as_matrix()
    y = df[target_name].as_matrix()

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create a list of alphas to cross-validate against
    alphas = np.logspace(-12, -0.5, 400)

    # Instantiate the linear model and visualizer
    model = LassoCV(alphas=alphas)
    visualizer = AlphaSelection(model)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    g = visualizer.poof(outpath="images/alpha_selection.png")             # Draw/show/poof the data
