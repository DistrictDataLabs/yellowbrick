import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split

from yellowbrick.classifier import ROCAUC


def load_occupancy():
    # Load the binary classification data set
    room = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

    features = ["temperature", "relative humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']

    # Extract the numpy arrays from the data frame
    X = room[features].values
    y = room.occupancy.values

    return X, y, classes


def load_game():
    # Load multi-class classification dataset
    game = pd.read_csv('../../../examples/data/game/game.csv')

    classes = ["win", "loss", "draw"]
    game.replace({'loss':-1, 'draw':0, 'win':1, 'x':2, 'o':3, 'b':4}, inplace=True)

    # Extract the numpy arrays from the data frame
    X = game.iloc[:, game.columns != 'outcome']
    y = game['outcome']

    return X, y, classes


def rocauc(X, y, model, outpath, **kwargs):
    # Create a new figure and axes
    _, ax = plt.subplots()

    # Instantiate the classification model and visualizer
    visualizer = ROCAUC(model, ax=ax, **kwargs)

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)

    # Save to disk
    visualizer.poof(outpath=outpath)

if __name__ == '__main__':

    # Occupancy data visualization
    X, y, classes = load_occupancy()

    # Draw the binary rocauc
    rocauc(
        X, y, LogisticRegression(), "images/rocauc_binary.png", classes=classes
    )

    # Draw a single binary decision curve
    rocauc(
        X, y, LinearSVC(), "images/rocauc_binary.png",
        micro=False, macro=False, per_class=False
    )

    # Game data visualization
    X, y, classes = load_game()

    # Draw the multiclass roc_auc
    rocauc(
        X, y, RidgeClassifier(), "images/rocauc_multiclass.png", classes=classes
    )
