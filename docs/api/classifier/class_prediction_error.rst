.. -*- mode: rst -*-

Class Prediction Error
======================

The Yellowbrick ``ClassPredictionError`` plot is a twist on other and sometimes more familiar classification model diagnostic tools like the :doc:`confusion_matrix` and :doc:`classification_report`. Like the :doc:`classification_report`, this plot shows the support (number of training samples) for each class in the fitted classification model as a stacked bar chart. Each bar is segmented to show the proportion of predictions (including false negatives and false positives, like a :doc:`confusion_matrix`) for each class. You can use a ``ClassPredictionError`` to visualize which classes your classifier is having a particularly difficult time with, and more importantly, what incorrect answers it is giving on a per-class basis. This can often enable you to better understand strengths and weaknesses of different models and particular challenges unique to your dataset.

The class prediction error chart provides a way to quickly understand how good your classifier is at predicting the right classes.

=================   ==============================
Visualizer           :class:`~yellowbrick.classifier.class_prediction_error.ClassPredictionError`
Quick Method         :func:`~yellowbrick.classifier.class_prediction_error.class_prediction_error`
Models               Classification
Workflow             Model evaluation
=================   ==============================

.. plot::
    :context: close-figs
    :alt:  Class Prediction Error plot on Fruit

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
        RandomForestClassifier(random_state=42, n_estimators=10), classes=classes
    )

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    visualizer.show()

In the above example, while the ``RandomForestClassifier`` appears to be fairly good at correctly predicting apples based on the features of the fruit, it often incorrectly labels pears as kiwis and mistakes kiwis for bananas.

By contrast, in the following example, the ``RandomForestClassifier`` does a great job at correctly predicting accounts in default, but it is a bit of a coin toss in predicting account holders who stayed current on bills.

.. plot::
    :context: close-figs
    :alt: Class Prediction Error on account standing

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
        RandomForestClassifier(n_estimators=10), classes=classes
    )

    # Fit the training data to the visualizer
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    visualizer.show()

Quick Method
------------

Similar functionality as above can be achieved in one line using the associated quick method, ``class_prediction_error``. This method will instantiate and fit a ``ClassPredictionError`` visualizer on the training data, then will score it on the optionally provided test data (or the training data if it is not provided).

.. plot::
    :context: close-figs
    :alt: class_prediction_error quick method

    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split as tts
    from yellowbrick.classifier import class_prediction_error
    from yellowbrick.datasets import load_occupancy

    # Load the dataset and split into train/test splits
    X, y = load_occupancy()
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, shuffle=True
    )

    class_prediction_error(
        LinearSVC(random_state=42),
        X_train, y_train, X_test, y_test,
        classes=["vacant", "occupied"]
    )


API Reference
-------------

.. automodule:: yellowbrick.classifier.class_prediction_error
    :members: ClassPredictionError, class_prediction_error
    :undoc-members:
    :show-inheritance:
