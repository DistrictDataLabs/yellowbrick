import os
import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features.importances import FeatureImportances
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso


DATA_DIR = os.path.relpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "examples", "data"
))


def feature_importances_(outpath):
    occupancy = pd.read_csv(os.path.join(DATA_DIR, "occupancy", "occupancy.csv"))

    feats = [
        "temperature", "relative humidity", "light", "C02", "humidity"
    ]

    X = occupancy[feats]
    y = occupancy['occupancy'].astype(int)

    fig = plt.figure()
    ax = fig.add_subplot()

    viz = FeatureImportances(GradientBoostingClassifier(), ax=ax)
    viz.fit(X, y)
    viz.poof(outpath=outpath)


def coef_(outpath):
    concrete = pd.read_csv(os.path.join(DATA_DIR, "concrete", "concrete.csv"))

    feats = ['cement','slag','ash','water','splast','coarse','fine','age']
    X = concrete[feats]
    y = concrete['strength']

    fig = plt.figure()
    ax = fig.add_subplot()

    feats = list(map(lambda s: s.title(), feats))
    viz = FeatureImportances(Lasso(), ax=ax, labels=feats, relative=False)
    viz.fit(X, y)
    viz.poof(outpath=outpath)


if __name__ == '__main__':
    feature_importances_("images/feature_importances.png")
    coef_("images/feature_importances_coef.png")
