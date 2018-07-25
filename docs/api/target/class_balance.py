# class_balance
# Generates images for the class balance documentation.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Thu Jul 19 12:09:40 2018 -0400
#
# ID: class_balance.py [] benjamin@bengfort.com $

"""
Generates images for the class balance documentation.
"""

##########################################################################
## Imports
##########################################################################

from yellowbrick.target import ClassBalance
from yellowbrick.datasets import load_occupancy, load_game

from sklearn.model_selection import train_test_split

def compare_class_balance(path="images/class_balance_compare.png"):
    data = load_occupancy()

    features = ["temperature", "relative_humidity", "light", "C02", "humidity"]
    classes = ['unoccupied', 'occupied']

    # Extract the numpy arrays from the data frame
    X = data[features]
    y = data["occupancy"]

    # Create the train and test data
    _, _, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Instantiate the classification model and visualizer
    visualizer = ClassBalance(labels=classes)

    visualizer.fit(y_train, y_test)
    return visualizer.poof(outpath=path)


def balance_class_balance(path="images/class_balance.png"):
    data = load_game()
    y = data["outcome"]

    oz = ClassBalance(labels=["draw", "loss", "win"])
    oz.fit(y)
    return oz.poof(outpath=path)



if __name__ == '__main__':
    compare_class_balance()
    balance_class_balance()
