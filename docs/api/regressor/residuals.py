import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from yellowbrick.regressor import ResidualsPlot


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

    # Instantiate the linear model and visualizer
    ridge = Ridge()
    visualizer = ResidualsPlot(ridge)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    g = visualizer.poof(outpath="images/residuals.png")             # Draw/show/poof the data
