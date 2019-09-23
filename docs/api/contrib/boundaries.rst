.. -*- mode: rst -*-

DecisionBoundaries Vizualizer
=============================

The DecisionBoundariesVisualizer is a bivariate data visualization algorithm that plots the decision boundaries of each class.

.. plot::
    :context: close-figs
    :alt: DecisionBoundariesVisualizer Nearest Neighbors

    from sklearn.model_selection import train_test_split as tts
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons
    from sklearn.neighbors import KNeighborsClassifier
    from yellowbrick.contrib.classifier import DecisionViz

    data_set = make_moons(noise=0.3, random_state=0)

    X, y = data_set
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=.4, random_state=42)

    viz = DecisionViz(
        KNeighborsClassifier(3), title="Nearest Neighbors",
        features=['Feature One', 'Feature Two'], classes=['A', 'B']
    )
    viz.fit(X_train, y_train)
    viz.draw(X_test, y_test)
    viz.show()


.. plot::
    :context: close-figs
    :alt: DecisionBoundariesVisualizer Linear SVM

    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split as tts
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons
    from sklearn.neighbors import KNeighborsClassifier
    from yellowbrick.contrib.classifier import DecisionViz

    data_set = make_moons(noise=0.3, random_state=0)

    X, y = data_set
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=.4, random_state=42)


    viz = DecisionViz(
        SVC(kernel="linear", C=0.025), title="Linear SVM",
        features=['Feature One', 'Feature Two'], classes=['A', 'B']
    )
    viz.fit(X_train, y_train)
    viz.draw(X_test, y_test)
    viz.show()


API Reference
-------------

.. automodule:: yellowbrick.contrib.classifier.boundaries
    :members: DecisionBoundariesVisualizer
    :undoc-members:
    :show-inheritance:
