.. -*- mode: rst -*-

Precision-Recall Curves
=======================

Precision-Recall curves are a metric used to evaluate a classifier's quality,
particularly when classes are very imbalanced. The precision-recall curve
shows the tradeoff between precision, a measure of result relevancy, and
recall, a measure of how many relevant results are returned. A large area
under the curve represents both high recall and precision, the best case
scenario for a classifier, showing a model that returns accurate results
for the majority of classes it selects.

=================   ==============================
Visualizer           :class:`~yellowbrick.classifier.prcurve.PrecisionRecallCurve`
Quick Method         :func:`~yellowbrick.classifier.prcurve.precision_recall_curve`
Models               Classification
Workflow             Model evaluation
=================   ==============================


Binary Classification
---------------------

.. plot::
    :context: close-figs
    :alt: PrecisionRecallCurve with Binary Classification

    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import train_test_split as tts
    from yellowbrick.classifier import PrecisionRecallCurve
    from yellowbrick.datasets import load_spam

    # Load the dataset and split into train/test splits
    X, y = load_spam()

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, shuffle=True)

    # Create the visualizer, fit, score, and show it
    viz = PrecisionRecallCurve(RidgeClassifier())
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()


The base case for precision-recall curves is the binary classification case, and this case is also the most visually interpretable. In the figure above we can see the precision plotted on the y-axis against the recall on the x-axis. The larger the filled in area, the stronger the classifier is. The red line annotates the *average precision*, a summary of the entire plot computed as the weighted average of precision achieved at each threshold such that the weight is the difference in recall from the previous threshold.

Multi-Label Classification
--------------------------

To support multi-label classification, the estimator is wrapped in a `OneVsRestClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_ to produce binary comparisons for each class (e.g. the positive case is the class and the negative case is any other class). The Precision-Recall curve is then computed as the micro-average of the precision and recall for all classes:

.. plot::
    :context: close-figs
    :alt: PrecisionRecallCurves with Multi-label Classification

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from sklearn.model_selection import train_test_split as tts
    from yellowbrick.classifier import PrecisionRecallCurve
    from yellowbrick.datasets import load_game

    # Load dataset and encode categorical variables
    X, y = load_game()
    X = OrdinalEncoder().fit_transform(X)
    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, shuffle=True)

    # Create the visualizer, fit, score, and show it
    viz = PrecisionRecallCurve(RandomForestClassifier(n_estimators=10))
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()


A more complex Precision-Recall curve can be computed, however, displaying the each curve individually, along with F1-score ISO curves (e.g. that show the relationship between precision and recall for various F1 scores).

.. plot::
    :context: close-figs
    :alt: PrecisionRecallCurves displaying each curve individually

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from sklearn.model_selection import train_test_split as tts
    from yellowbrick.classifier import PrecisionRecallCurve
    from yellowbrick.datasets import load_game

    # Load dataset and encode categorical variables
    X, y = load_game()
    X = OrdinalEncoder().fit_transform(X)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, shuffle=True)

    # Create the visualizer, fit, score, and show it
    viz = PrecisionRecallCurve(
        MultinomialNB(), per_class=True, iso_f1_curves=True,
        fill_area=False, micro=False, classes=encoder.classes_
    )
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()


.. seealso:: `Scikit-Learn: Model Selection with Precision Recall Curves <http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html>`_


Quick Method
------------

Similar functionality as above can be achieved in one line using the associated quick method, ``precision_recall_curve``. This method will instantiate and fit a ``PrecisionRecallCurve`` visualizer on the training data, then will score it on the optionally provided test data (or the training data if it is not provided).

.. plot::
    :context: close-figs
    :alt: precision_recall_curve quick method with binary classification

    from sklearn.naive_bayes import BernoulliNB
    from sklearn.model_selection import train_test_split as tts
    from yellowbrick.classifier import precision_recall_curve
    from yellowbrick.datasets import load_spam

    # Load the dataset and split into train/test splits
    X, y = load_spam()

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, shuffle=True)

    # Create the visualizer, fit, score, and show it
    viz = precision_recall_curve(BernoulliNB(), X_train, y_train, X_test, y_test)


API Reference
-------------

.. automodule:: yellowbrick.classifier.prcurve
    :members: PrecisionRecallCurve, precision_recall_curve
    :undoc-members:
    :show-inheritance:
