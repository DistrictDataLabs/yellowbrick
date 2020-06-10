.. -*- mode: rst -*-

Precision-Recall Curves
=======================

The ``PrecisionRecallCurve`` shows the tradeoff between a classifier's precision, a measure of result relevancy, and recall, a measure of completeness. For each class, precision is defined as the ratio of true positives to the sum of true and false positives, and recall is the ratio of true positives to the sum of true positives and false negatives.

=================   ==============================
Visualizer           :class:`~yellowbrick.classifier.prcurve.PrecisionRecallCurve`
Quick Method         :func:`~yellowbrick.classifier.prcurve.precision_recall_curve`
Models               Classification
Workflow             Model evaluation
=================   ==============================

**precision**
    Precision can be seen as a measure of a classifier's exactness. For each class, it is defined as the ratio of true positives to the sum of true and false positives. Said another way, "for all instances classified positive, what percent was correct?"

**recall**
    Recall is a measure of the classifier's completeness; the ability of a classifier to correctly find all positive instances. For each class, it is defined as the ratio of true positives to the sum of true positives and false negatives. Said another way, "for all instances that were actually positive, what percent was classified correctly?"

**average precision**
    Average precision expresses the precision-recall curve in a single number, which
    represents the area under the curve. It is computed as the weighted average of precision achieved at each threshold, where the weights are the differences in recall from the previous thresholds.

Both precision and recall vary between 0 and 1, and in our efforts to select and tune machine learning models, our goal is often to try to maximize both precision and recall, i.e. a model that returns accurate results for the majority of classes it selects. This would result in a ``PrecisionRecallCurve`` visualization with a high area under the curve.

Binary Classification
---------------------

The base case for precision-recall curves is the binary classification case, and this case is also the most visually interpretable. In the figure below we can see the precision plotted on the y-axis against the recall on the x-axis. The larger the filled in area, the stronger the classifier. The red line annotates the average precision.

.. plot::
    :context: close-figs
    :alt: PrecisionRecallCurve with Binary Classification

    import matplotlib.pyplot as plt

    from yellowbrick.datasets import load_spam
    from sklearn.linear_model import RidgeClassifier
    from yellowbrick.classifier import PrecisionRecallCurve
    from sklearn.model_selection import train_test_split as tts

    # Load the dataset and split into train/test splits
    X, y = load_spam()

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, shuffle=True, random_state=0
    )

    # Create the visualizer, fit, score, and show it
    viz = PrecisionRecallCurve(RidgeClassifier(random_state=0))
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()

One way to use ``PrecisionRecallCurves`` is for model comparison, by examining which have the highest average precision. For instance, the below visualization suggest that a ``LogisticRegression`` model might be better than a ``RidgeClassifier`` for this particular dataset:

.. plot::
    :context: close-figs
    :include-source: False
    :alt: Comparing PrecisionRecallCurves with Binary Classification

    import matplotlib.pyplot as plt

    from yellowbrick.datasets import load_spam
    from yellowbrick.classifier import PrecisionRecallCurve
    from sklearn.model_selection import train_test_split as tts
    from sklearn.linear_model import RidgeClassifier, LogisticRegression

    # Load the dataset and split into train/test splits
    X, y = load_spam()

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, shuffle=True, random_state=0
    )

    # Create the visualizers, fit, score, and show them
    models = [
        RidgeClassifier(random_state=0), LogisticRegression(random_state=0)
    ]
    _, axes = plt.subplots(ncols=2, figsize=(8,4))

    for idx, ax in enumerate(axes.flatten()):
        viz = PrecisionRecallCurve(models[idx], ax=ax, show=False)
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()

    plt.show()

Precision-recall curves are one of the methods used to evaluate a classifier's quality, particularly when classes are very imbalanced. The below plot suggests that our classifier improves when we increase the weight of the "spam" case (which is 1), and decrease the weight for the "not spam" case (which is 0).

.. plot::
    :context: close-figs
    :alt: Optimizing PrecisionRecallCurve with Binary Classification

    from yellowbrick.datasets import load_spam
    from sklearn.linear_model import LogisticRegression
    from yellowbrick.classifier import PrecisionRecallCurve
    from sklearn.model_selection import train_test_split as tts

    # Load the dataset and split into train/test splits
    X, y = load_spam()

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.2, shuffle=True, random_state=0
    )

    # Specify class weights to shift the threshold towards spam classification
    weights = {0:0.2, 1:0.8}

    # Create the visualizer, fit, score, and show it
    viz = PrecisionRecallCurve(
        LogisticRegression(class_weight=weights, random_state=0)
    )
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()

Multi-Label Classification
--------------------------

To support multi-label classification, the estimator is wrapped in a `OneVsRestClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_ to produce binary comparisons for each class (e.g. the positive case is the class and the negative case is any other class). The precision-recall curve can then be computed as the micro-average of the precision and recall for all classes (by setting ``micro=True``), or individual curves can be plotted for each class (by setting ``per_class=True``):

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
    viz = PrecisionRecallCurve(
        RandomForestClassifier(n_estimators=10),
        per_class=True,
        cmap="Set1"
    )
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

    # Encode the target (we'll use the encoder to retrieve the class labels)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, shuffle=True)

    # Create the visualizer, fit, score, and show it
    viz = PrecisionRecallCurve(
        MultinomialNB(),
        classes=encoder.classes_,
        colors=["purple", "cyan", "blue"],
        iso_f1_curves=True,
        per_class=True,
        micro=False
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
