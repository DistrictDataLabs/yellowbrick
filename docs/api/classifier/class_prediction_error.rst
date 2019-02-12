.. -*- mode: rst -*-

Class Prediction Error
======================
The class prediction error chart provides a way to quickly understand how good your classifier is at predicting the right classes.
The Yellowbrick `ClassPredictionError` plot is a twist on other and sometimes more familiar classification model diagnostic tools like `ConfusionMatrix` and `ClassificationReport`. Like the `ClassificationReport`, this plot shows the support (number of training samples) for each class in the fitted classification model as a stacked bar chart. Each bar is segmented to show the proportion of predictions (including false negatives and false positives, like a `ConfusionMatrix`) for each class. You can use a `ClassPredictionError` to visualize which classes your classifier is having a particularly difficult time with, and more importantly, what incorrect answers it is giving on a per-class basis. This can often enable you to better understand strengths and weaknesses of different models and particular challenges unique to your dataset.

In this example, while the `RandomForestClassifier` appears to be fairly good at correctly predicting apples based on the features of the fruit, it often incorrectly labels pears as oranges and mistakes kiwis for pears.

.. plot::
    :context: close-figs

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from yellowbrick.classifier import ClassPredictionError


    # Create classification dataset
    X, y = make_classification(
        n_samples=1000, n_classes=5, n_informative=3, n_clusters_per_class=1,
        random_state=36,
    )

    classes = ["apple", "kiwi", "pear", "banana", "orange"]

    # Perform 80/20 training/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        random_state=42)
    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(
        RandomForestClassifier(random_state=42), classes=classes
    )

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    visualizer.poof()

In this example again, the `RandomForestClassifier` does a great job at correctly predicting accounts in default while it is a toss of a coin in predicting if they are current on bills.

.. plot::
    :context: close-figs

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from yellowbrick.classifier import ClassPredictionError
    from yellowbrick.datasets import load_credit

    X, y = load_credit()
    
    classes = ['account in default', 'current with bills']

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
