import pandas as pd
import matplotlib.pyplot as plt

from yellowbrick.features.importances import FeatureImportances
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso

from tests.dataset import DatasetMixin

loader = DatasetMixin()


def feature_importances_(outpath):
    occupancy = pd.DataFrame(loader.load_data('occupancy'))

    feats = [
        "temperature", "relative_humidity", "light", "C02", "humidity"
    ]

    X = occupancy[feats]
    y = occupancy['occupancy'].astype(int)

    fig = plt.figure()
    ax = fig.add_subplot()

    viz = FeatureImportances(GradientBoostingClassifier(), ax=ax)
    viz.fit(X, y)
    viz.poof(outpath=outpath)


def coef_(outpath):
    concrete = pd.DataFrame(loader.load_data('concrete'))
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
    feature_importances_("images/feature_importances_.png")
    coef_("images/coef_.png")
