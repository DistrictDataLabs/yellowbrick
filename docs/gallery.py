#!/usr/bin/env python3
# Generates images for the gallery

import os
import argparse
import numpy as np
import os.path as path
import matplotlib.pyplot as plt

from yellowbrick.datasets import load_occupancy, load_credit, load_concrete
from yellowbrick.datasets import load_spam, load_game, load_energy, load_hobbies

from yellowbrick.model_selection import RFECV, FeatureImportances

from yellowbrick.features import PCA, Manifold, JointPlot
from yellowbrick.features import RadViz, Rank1D, Rank2D, ParallelCoordinates

from yellowbrick.contrib.scatter import ScatterVisualizer

from yellowbrick.regressor import ResidualsPlot, PredictionError, AlphaSelection

from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix
from yellowbrick.classifier import ROCAUC, PRCurve, ClassPredictionError

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance

from yellowbrick.model_selection import ValidationCurve, LearningCurve, CVScores
from yellowbrick.contrib.classifier import DecisionViz

from yellowbrick.text import (
    FreqDistVisualizer,
    TSNEVisualizer,
    DispersionPlot,
    PosTagVisualizer,
)

from yellowbrick.target import (
    BalancedBinningReference,
    ClassBalance,
    FeatureCorrelation,
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, LassoCV, RidgeCV
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.datasets import make_classification, make_blobs, make_moons
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


GALLERY = path.join(path.dirname(__file__), "images", "gallery")


##########################################################################
## Helper Methods
##########################################################################


def newfig():
    """
    Helper function to create an axes object of the gallery dimensions.
    """
    # NOTE: this figsize generates a better thumbnail
    _, ax = plt.subplots(figsize=(8, 4))
    return ax


def savefig(viz, name, gallery=GALLERY):
    """
    Saves the figure to the gallery directory
    """
    if not path.exists(gallery):
        os.makedirs(gallery)

    # Must save as png
    if len(name.split(".")) > 1:
        raise ValueError("name should not specify extension")

    outpath = path.join(gallery, name + ".png")
    viz.show(outpath=outpath)
    print("created {}".format(outpath))


##########################################################################
## Feature Analysis
##########################################################################


def radviz():
    X, y = load_occupancy()
    oz = RadViz(ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "radviz")


def rank1d():
    X, y = load_credit()
    oz = Rank1D(algorithm="shapiro", ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "rank1d_shapiro")


def rank2d():
    X, y = load_credit()
    oz = Rank2D(algorithm="covariance", ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "rank2d_covariance")


def pcoords():
    X, y = load_occupancy()
    oz = ParallelCoordinates(sample=0.05, shuffle=True, ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "parallel_coordinates")


def pca():
    X, y = load_credit()
    colors = np.array(["r" if yi else "b" for yi in y])
    oz = PCA(scale=True, color=colors, proj_dim=3)
    oz.fit_transform(X, y)
    savefig(oz, "pca_projection_3d")


def manifold(dataset, manifold):
    if dataset == "concrete":
        X, y = load_concrete()
    elif dataset == "occupancy":
        X, y = load_occupancy()
    else:
        raise ValueError("unknown dataset")

    oz = Manifold(manifold=manifold, ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "{}_{}_manifold".format(dataset, manifold))


def scatter():
    X, y = load_occupancy()
    oz = ScatterVisualizer(x="light", y="CO2", ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "scatter")


def jointplot():
    X, y = load_concrete()
    oz = JointPlot(columns=["cement", "splast"], ax=newfig())
    oz.fit_transform(X, y)
    savefig(oz, "jointplot")


##########################################################################
## Regression
##########################################################################


def residuals():
    X, y = load_concrete()
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    oz = ResidualsPlot(Ridge(), ax=newfig())
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    savefig(oz, "residuals")


def peplot():
    X, y = load_concrete()
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    oz = PredictionError(Lasso(), ax=newfig())
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    savefig(oz, "prediction_error")


def alphas():
    X, y = load_concrete()
    alphas = np.logspace(-10, 1, 400)
    oz = AlphaSelection(LassoCV(alphas=alphas), ax=newfig())
    oz.fit(X, y)
    savefig(oz, "alpha_selection")


##########################################################################
## Classification
##########################################################################


def classreport():
    X, y = load_occupancy()
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    oz = ClassificationReport(GaussianNB(), support=True, ax=newfig())
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    savefig(oz, "classification_report")


def confusion(dataset):
    if dataset == "iris":
        data = load_iris()
    elif dataset == "digits":
        data = load_digits()
    else:
        raise ValueError("uknown dataset")

    X_train, X_test, y_train, y_test = tts(data.data, data.target, test_size=0.2)
    oz = ConfusionMatrix(LogisticRegression(), ax=newfig())
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    savefig(oz, "confusion_matrix_{}".format(dataset))


def rocauc(dataset):
    if dataset == "binary":
        X, y = load_occupancy()
        model = GaussianNB()
    elif dataset == "multiclass":
        X, y = load_game()
        X = OrdinalEncoder().fit_transform(X)
        model = RidgeClassifier()
    else:
        raise ValueError("uknown dataset")

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    oz = ROCAUC(model, ax=newfig())
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    savefig(oz, "rocauc_{}".format(dataset))


def prcurve(dataset):
    if dataset == "binary":
        X, y = load_spam()
        model = RidgeClassifier()
        kws = {}
    elif dataset == "multiclass":
        X, y = load_game()
        X = OrdinalEncoder().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
        model = MultinomialNB()
        kws = {
            "per_class": True,
            "iso_f1_curves": True,
            "fill_area": False,
            "micro": False,
        }
    else:
        raise ValueError("uknown dataset")

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, shuffle=True)
    oz = PRCurve(model, ax=newfig(), **kws)
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    savefig(oz, "precision_recall_{}".format(dataset))


def classprede():
    X, y = make_classification(
        n_samples=1000, n_classes=5, n_informative=3, n_clusters_per_class=1
    )

    classes = ["apple", "kiwi", "pear", "banana", "orange"]

    # Perform 80/20 training/test split
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.20)
    oz = ClassPredictionError(RandomForestClassifier(), classes=classes, ax=newfig())
    oz.fit(X_train, y_train)
    oz.score(X_test, y_test)
    savefig(oz, "class_prediction_error")


def discrimination():
    X, y = load_spam()
    oz = DiscriminationThreshold(LogisticRegression(solver="lbfgs"), ax=newfig())
    oz.fit(X, y)
    savefig(oz, "discrimination_threshold")


##########################################################################
## Clustering
##########################################################################


def elbow():
    X, _ = make_blobs(centers=8, n_features=12, shuffle=True)
    oz = KElbowVisualizer(KMeans(), k=(4, 12), ax=newfig())
    oz.fit(X)
    savefig(oz, "elbow")


def silhouette():
    X, _ = make_blobs(centers=8)
    oz = SilhouetteVisualizer(MiniBatchKMeans(6), ax=newfig())
    oz.fit(X)
    savefig(oz, "silhouette")


def icdm():
    X, _ = make_blobs(centers=12, n_samples=1000, n_features=16, shuffle=True)
    oz = InterclusterDistance(KMeans(9), ax=newfig())
    oz.fit(X)
    savefig(oz, "icdm")


##########################################################################
## Model Selection
##########################################################################


def validation():
    X, y = load_energy()
    oz = ValidationCurve(
        DecisionTreeRegressor(),
        param_name="max_depth",
        param_range=np.arange(1, 11),
        cv=10,
        scoring="r2",
        ax=newfig(),
    )
    oz.fit(X, y)
    savefig(oz, "validation_curve")


def learning():
    X, y = load_energy()
    sizes = np.linspace(0.3, 1.0, 10)
    oz = LearningCurve(RidgeCV(), train_sizes=sizes, scoring="r2", ax=newfig())
    oz.fit(X, y)
    savefig(oz, "learning_curve")


def cvscores():
    X, y = load_energy()
    oz = CVScores(Ridge(), scoring="r2", cv=10, ax=newfig())
    oz.fit(X, y)
    savefig(oz, "cv_scores")


def decision():
    X, y = make_moons(noise=0.3)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.20)

    oz = DecisionViz(KNeighborsClassifier(3), ax=newfig())
    oz.fit(X_train, y_train)
    oz.draw(X_test, y_test)
    savefig(oz, "decision_boundaries")


def importances():
    X, y = load_occupancy()
    oz = FeatureImportances(RandomForestClassifier(), ax=newfig())
    oz.fit(X, y)
    savefig(oz, "feature_importances")


def rfecv():
    X, y = load_credit()
    model = RandomForestClassifier(n_estimators=10)
    oz = RFECV(model, cv=3, scoring="f1_weighted", ax=newfig())
    oz.fit(X, y)
    savefig(oz, "rfecv_sklearn_example")


##########################################################################
## Text Model Diagnostics
##########################################################################


def freqdist():
    corpus = load_hobbies()
    vecs = CountVectorizer()
    docs = vecs.fit_transform(corpus.data)

    oz = FreqDistVisualizer(features=vecs.get_feature_names(), ax=newfig())
    oz.fit(docs)
    savefig(oz, "freqdist")


def tsne():
    corpus = load_hobbies()
    docs = TfidfVectorizer().fit_transform(corpus.data)

    oz = TSNEVisualizer(ax=newfig())
    oz.fit(docs, corpus.target)
    savefig(oz, "corpus_tsne")


def dispersion():
    corpus = load_hobbies()
    target_words = ["Game", "player", "score", "oil", "Man"]

    oz = DispersionPlot(target_words, ax=newfig())
    oz.fit([doc.split() for doc in corpus.data])
    savefig(oz, "dispersion")


def postag():
    tagged_stanzas = [
        [
            [
                ("Whose", "JJ"),
                ("woods", "NNS"),
                ("these", "DT"),
                ("are", "VBP"),
                ("I", "PRP"),
                ("think", "VBP"),
                ("I", "PRP"),
                ("know", "VBP"),
                (".", "."),
            ],
            [
                ("His", "PRP$"),
                ("house", "NN"),
                ("is", "VBZ"),
                ("in", "IN"),
                ("the", "DT"),
                ("village", "NN"),
                ("though", "IN"),
                (";", ":"),
                ("He", "PRP"),
                ("will", "MD"),
                ("not", "RB"),
                ("see", "VB"),
                ("me", "PRP"),
                ("stopping", "VBG"),
                ("here", "RB"),
                ("To", "TO"),
                ("watch", "VB"),
                ("his", "PRP$"),
                ("woods", "NNS"),
                ("fill", "VB"),
                ("up", "RP"),
                ("with", "IN"),
                ("snow", "NNS"),
                (".", "."),
            ],
        ],
        [
            [
                ("My", "PRP$"),
                ("little", "JJ"),
                ("horse", "NN"),
                ("must", "MD"),
                ("think", "VB"),
                ("it", "PRP"),
                ("queer", "JJR"),
                ("To", "TO"),
                ("stop", "VB"),
                ("without", "IN"),
                ("a", "DT"),
                ("farmhouse", "NN"),
                ("near", "IN"),
                ("Between", "NNP"),
                ("the", "DT"),
                ("woods", "NNS"),
                ("and", "CC"),
                ("frozen", "JJ"),
                ("lake", "VB"),
                ("The", "DT"),
                ("darkest", "JJS"),
                ("evening", "NN"),
                ("of", "IN"),
                ("the", "DT"),
                ("year", "NN"),
                (".", "."),
            ]
        ],
        [
            [
                ("He", "PRP"),
                ("gives", "VBZ"),
                ("his", "PRP$"),
                ("harness", "NN"),
                ("bells", "VBZ"),
                ("a", "DT"),
                ("shake", "NN"),
                ("To", "TO"),
                ("ask", "VB"),
                ("if", "IN"),
                ("there", "EX"),
                ("is", "VBZ"),
                ("some", "DT"),
                ("mistake", "NN"),
                (".", "."),
            ],
            [
                ("The", "DT"),
                ("only", "JJ"),
                ("other", "JJ"),
                ("sound", "NN"),
                ("’", "NNP"),
                ("s", "VBZ"),
                ("the", "DT"),
                ("sweep", "NN"),
                ("Of", "IN"),
                ("easy", "JJ"),
                ("wind", "NN"),
                ("and", "CC"),
                ("downy", "JJ"),
                ("flake", "NN"),
                (".", "."),
            ],
        ],
        [
            [
                ("The", "DT"),
                ("woods", "NNS"),
                ("are", "VBP"),
                ("lovely", "RB"),
                (",", ","),
                ("dark", "JJ"),
                ("and", "CC"),
                ("deep", "JJ"),
                (",", ","),
                ("But", "CC"),
                ("I", "PRP"),
                ("have", "VBP"),
                ("promises", "NNS"),
                ("to", "TO"),
                ("keep", "VB"),
                (",", ","),
                ("And", "CC"),
                ("miles", "NNS"),
                ("to", "TO"),
                ("go", "VB"),
                ("before", "IN"),
                ("I", "PRP"),
                ("sleep", "VBP"),
                (",", ","),
                ("And", "CC"),
                ("miles", "NNS"),
                ("to", "TO"),
                ("go", "VB"),
                ("before", "IN"),
                ("I", "PRP"),
                ("sleep", "VBP"),
                (".", "."),
            ]
        ],
    ]
    oz = PosTagVisualizer(ax=newfig())
    oz.fit(tagged_stanzas)
    savefig(oz, "postag")


##########################################################################
## Target Visualizations
##########################################################################


def binning():
    _, y = load_concrete()
    oz = BalancedBinningReference(ax=newfig())
    oz.fit(y)
    savefig(oz, "balanced_binning_reference")


def balance():
    X, y = load_occupancy()
    _, _, y_train, y_test = tts(X, y, test_size=0.2)

    oz = ClassBalance(ax=newfig(), labels=["unoccupied", "occupied"])
    oz.fit(y_train, y_test)
    savefig(oz, "class_balance")


def featcorr():
    data = load_diabetes()

    oz = FeatureCorrelation(ax=newfig())
    oz.fit(data.data, data.target)
    savefig(oz, "feature_correlation")


##########################################################################
## Main Method
##########################################################################

if __name__ == "__main__":
    plots = {
        "all": None,
        "radviz": radviz,
        "rank1d": rank1d,
        "rank2d": rank2d,
        "pcoords": pcoords,
        "pca": pca,
        "concrete_tsne": lambda: manifold("concrete", "tsne"),
        "occupancy_tsne": lambda: manifold("occupancy", "tsne"),
        "concrete_isomap": lambda: manifold("concrete", "isomap"),
        "importances": importances,
        "rfecv": rfecv,
        "scatter": scatter,
        "jointplot": jointplot,
        "residuals": residuals,
        "peplot": peplot,
        "alphas": alphas,
        "classreport": classreport,
        "confusion_digits": lambda: confusion("digits"),
        "confusion_iris": lambda: confusion("iris"),
        "rocauc_binary": lambda: rocauc("binary"),
        "rocauc_multi": lambda: rocauc("multiclass"),
        "prcurve_binary": lambda: prcurve("binary"),
        "prcurve_multi": lambda: prcurve("multiclass"),
        "classprede": classprede,
        "discrimination": discrimination,
        "elbow": elbow,
        "silhouette": silhouette,
        "icdm": icdm,
        "validation": validation,
        "learning": learning,
        "cvscores": cvscores,
        "freqdist": freqdist,
        "tsne": tsne,
        "dispersion": dispersion,
        "postag": postag,
        "decision": decision,
        "binning": binning,
        "balance": balance,
        "featcorr": featcorr,
    }

    parser = argparse.ArgumentParser(description="gallery image generator")
    parser.add_argument(
        "plots",
        nargs="+",
        choices=plots.keys(),
        metavar="plot",
        help="names of images to generate",
    )
    args = parser.parse_args()

    queue = frozenset(args.plots)
    if "all" in queue:
        queue = frozenset(plots.keys())

    for item in queue:
        method = plots[item]
        if method is not None:
            method()
