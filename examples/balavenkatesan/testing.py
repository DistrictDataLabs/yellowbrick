import pandas as pd
import csv
from sklearn import neighbors
from sklearn import datasets
import numpy as np
import yellowbrick as yb
from yellowbrick.neighbors import KnnDecisionBoundariesVisualizer
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def load_adm_sat_school_data(return_X_y=False):

    with open("./merged_adm_sat_data.csv") as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])

    df = pd.read_csv(
        "./merged_adm_sat_data.csv", sep=",", usecols=(0, 1, 2, 3), skiprows=0
    )
    data = np.empty((n_samples, n_features), dtype=int)
    target = np.ma.empty((n_samples,), dtype=int)

    for index, row in df.iterrows():
        data[index] = np.asarray(
            [df.iloc[index][0], df.iloc[index][1], df.iloc[index][2]], dtype=np.float
        )
        target[index] = np.asarray(df.iloc[index][3], dtype=np.int)

    feature_names = np.array(["ACT_AVG", "SAT_AVG", "GRAD_DEBT", "REGION"])

    if return_X_y:
        return data, target

    return datasets.base.Bunch(
        data=data,
        target=target,
        target_names=target_names,
        DESCR="School Data set",
        feature_names=feature_names,
    )


def show_plot(X, y, n_neighbors=10, h=0.2):
    # Create color maps
    cmap_light = ListedColormap(
        [
            "#FFAAAA",
            "#AAFFAA",
            "#AAAAFF",
            "#FFAAAA",
            "#AAFFAA",
            "#AAAAFF",
            "#FFAAAA",
            "#AAFFAA",
            "#AAAAFF",
            "#AAAAFF",
        ]
    )
    cmap_bold = ListedColormap(
        [
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FF0000",
            "#FF0000",
            "#FF0000",
            "#FF0000",
            "#FF0000",
            "#FF0000",
            "#FF0000",
        ]
    )

    for weights in ["uniform", "distance"]:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)
        clf.n_neighbors = n_neighbors

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(
            "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
        )

    plt.show()


if __name__ == "__main__":
    school = load_adm_sat_school_data()
    X = school.data[:, :2]  # we only take the first two features.
    y = school.target
    # show_plot(X,y,3)
    model = neighbors.KNeighborsClassifier(10)
    model.fit(X, y)
    model.predict(X)
    # visualizer = KnnDecisionBoundariesVisualizer(model, classes=school.target_names, features=school.feature_names[:2])
    visualizer = KnnDecisionBoundariesVisualizer(model)
    visualizer.fit_draw_show(X, y)
