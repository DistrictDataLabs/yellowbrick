.. -*- mode: rst -*-

Class Prediction Error
======================
The class prediction error chart provides a way to quickly understand how good your classifier is at predicting the right classes.

.. plot::
    :context: close-figs

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from yellowbrick.classifier import ClassPredictionError


    # Create classification dataset
    X, y = make_classification(
        n_samples=1000, n_classes=5, n_informative=3, n_clusters_per_class=1
    )

    classes = ["apple", "kiwi", "pear", "banana", "orange"]

    # Perform 80/20 training/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)
    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(
        RandomForestClassifier(), classes=classes
    )

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    visualizer.poof()


.. plot::
    :context: close-figs
    :include-source: False
    :nofigs:

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from yellowbrick.classifier import ClassPredictionError
    from yellowbrick.datasets import load_credit

    X, y = load_credit()
    
    classes = ['default', 'current']

    # Perform 80/20 training/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)
    
    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(
        RandomForestClassifier(), classes=classes
    )

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    visualizer.poof()

    
    
API Reference
-------------

.. automodule:: yellowbrick.classifier.class_prediction_error
    :members: ClassPredictionError
    :undoc-members:
    :show-inheritance:
