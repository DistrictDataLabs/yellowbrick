import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from yellowbrick.classifier import ROCAUC


if __name__ == '__main__':
    # Load the regression data set
    data = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']

    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.occupancy.as_matrix()

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Instantiate the classification model and visualizer
    logistic = LogisticRegression()
    visualizer = ROCAUC(logistic)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    g = visualizer.poof(outpath="images/rocauc.png")             # Draw/show/poof the data
