# classification_report
# Generates images for the classification report documentation.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sun Mar 18 16:35:30 2018 -0400
#
# ID: classification_report.py [] benjamin@bengfort.com $

"""
Generates images for the classification report documentation.
"""

##########################################################################
## Imports
##########################################################################

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split as tts

from yellowbrick.classifier import ClassificationReport


##########################################################################
## Quick Methods
##########################################################################

def make_dataset():
    data = pd.read_csv("../../../examples/data/occupancy/occupancy.csv")

    X = data[["temperature", "relative humidity", "light", "C02", "humidity"]]
    y = data.occupancy

    return tts(X, y, test_size=0.2)


def make_gb_report(path="images/classification_report.png"):
    X_train, X_test, y_train, y_test = make_dataset()

    _, ax = plt.subplots()

    bayes = GaussianNB()
    viz = ClassificationReport(bayes, ax=ax, classes=['unoccupied', 'occupied'])

    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)

    viz.poof(outpath=path)


##########################################################################
## Main Method
##########################################################################

if __name__ == '__main__':
    make_gb_report()
