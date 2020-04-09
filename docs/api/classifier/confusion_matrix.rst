.. -*- mode: rst -*-

Confusion Matrix
================

The ``ConfusionMatrix`` visualizer is a ``ScoreVisualizer`` that takes a
fitted scikit-learn classifier and a set of test ``X`` and ``y`` values and
returns a report showing how each of the test values predicted classes
compare to their actual classes. Data scientists use confusion matrices
to understand which classes are most easily confused. These provide
similar information as what is available in a ``ClassificationReport``, but
rather than top-level scores, they provide deeper insight into the
classification of individual data points.

Below are a few examples of using the ``ConfusionMatrix`` visualizer; more
information can be found by looking at the
scikit-learn documentation on `confusion matrices <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>`_.

=================   =================
Visualizer           :class:`~yellowbrick.classifier.confusion_matrix.ConfusionMatrix`
Quick Method         :func:`~yellowbrick.classifier.confusion_matrix.confusion_matrix`
Models               Classification
Workflow             Model evaluation
=================   =================


.. plot::
    :context: close-figs
    :alt: ConfusionMatrix plot of sklearn Digits dataset

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split as tts
    from sklearn.linear_model import LogisticRegression
    from yellowbrick.classifier import ConfusionMatrix

    # We'll use the handwritten digits data set from scikit-learn.
    # Each feature of this dataset is an 8x8 pixel image of a handwritten number.
    # Digits.data converts these 64 pixels into a single array of features
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = tts(X, y, test_size =0.2, random_state=11)

    model = LogisticRegression(multi_class="auto", solver="liblinear")

    # The ConfusionMatrix visualizer taxes a model
    cm = ConfusionMatrix(model, classes=[0,1,2,3,4,5,6,7,8,9])

    # Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
    cm.fit(X_train, y_train)

    # To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
    # and then creates the confusion_matrix from scikit-learn.
    cm.score(X_test, y_test)

    # How did we do?
    cm.show()


Plotting with Class Names
-------------------------

Class names can be added to a ``ConfusionMatrix`` plot using the ``label_encoder`` argument. The ``label_encoder`` can be a `sklearn.preprocessing.LabelEncoder <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_ (or anything with an ``inverse_transform`` method that performs the mapping), or a ``dict`` with the encoding-to-string mapping as in the example below:

.. plot::
    :context: close-figs
    :alt: ConfusionMatrix plot with class names

    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split as tts

    from yellowbrick.classifier import ConfusionMatrix

    iris = load_iris()
    X = iris.data
    y = iris.target
    classes = iris.target_names

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

    model = LogisticRegression(multi_class="auto", solver="liblinear")

    iris_cm = ConfusionMatrix(
        model, classes=classes,
        label_encoder={0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    )

    iris_cm.fit(X_train, y_train)
    iris_cm.score(X_test, y_test)
    iris_cm.show()

Quick Method
------------

The same functionality above can be achieved with the associated quick method ``confusion_matrix``. This method will build the ``ConfusionMatrix`` object with the associated arguments, fit it, then (optionally) immediately show it. In the below example we can see how a ``LogisticRegression`` struggles to effectively model the credit dataset (hint: check out :class:`~yellowbrick.features.rankd.Rank2D` to examine for multicollinearity!).

.. plot::
    :context: close-figs
    :alt: confusion_matrix on the credit dataset

    from yellowbrick.datasets import load_credit
    from yellowbrick.classifier import confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split as tts

    #Load the classification dataset
    X, y = load_credit()

    #Create the train and test data
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

    # Instantiate the visualizer with the classification model
    confusion_matrix(
        LogisticRegression(),
        X_train, y_train, X_test, y_test,
        classes=['not_defaulted', 'defaulted']
    )
    plt.tight_layout()


API Reference
-------------

.. automodule:: yellowbrick.classifier.confusion_matrix
    :members: ConfusionMatrix, confusion_matrix
    :undoc-members:
    :show-inheritance:
