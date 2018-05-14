.. -*- mode: rst -*-

Learning Curve
==============

A learning curve shows the relationship of the training score vs the cross validated test score for an estimator with a varying number of training samples. This visualization is typically used two show two things:

1. How much the estimator benefits from more data (e.g. do we have "enough data" or will the estimator get better if used in an online fashion).
2. If the estimator is more sensitive to error due to variance vs. error due to bias.

Consider the following learning curves (generated with Yellowbrick, but from `Plotting Learning Curves <http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html>`_ in the scikit-learn documentation):

.. image:: images/learning_curve_sklearn_example.png

If the training and cross validation scores converge together as more data is added (shown in the left figure), then the model will probably not benefit from more data. If the training score is much greater than the validation score (as showin in the right figure) then the model probably requires more training examples in order to generalize more effectively.

The curves are plotted with the mean scores, however variability during cross-validation is shown with the shaded areas that represent a standard deviation above and below the mean for all cross-validations. If the model suffers from error due to bias, then there will likely be more variability around the training score curve. If the model suffers from error due to variance, then there will be more variability around the cross validated score.

.. note:: Learning curves can be generated for all estimators that have ``fit()`` and ``predict()`` methods as well as a single scoring metric. This includes classifiers, regressors, and clustering as we will see in the following examples.

Classification
--------------

In the following example we show how to visualize the learning curve of a classification model. After loading a ``DataFrame`` and performing categorical encoding, we create a ``StratifiedKFold`` cross-validation strategy to ensure all of our classes in each split are represented with the same proportion. We then fit the visualizer using the ``f1_weighted`` scoring metric as opposed to the default metric, accuracy, to get a better sense of the relationship of precision and recall in our classifier.

.. code:: python

    import numpy as np

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import StratifiedKFold
    from yellowbrick.model_selection import LearningCurve

    # Load a classification data set
    data = load_data('game')

    # Specify the features of interest and the target
    target = "outcome"
    features = [col for col in data.columns if col != target]

    # Encode the categorical data with one-hot encoding
    X = pd.get_dummies(data[features])
    y = data[target]

    # Create the learning curve visualizer
    cv = StratifiedKFold(12)
    sizes = np.linspace(0.3, 1.0, 10)

    viz = LearningCurve(
        MultinomialNB(), cv=cv, train_sizes=sizes,
        scoring='f1_weighted', n_jobs=4
    )

    # Fit and poof the visualizer
    viz.fit(X, y)
    viz.poof()

.. image:: images/learning_curve_classifier.png

This learning curve shows high test variability and a low score up to around 30,000 instances, however after this level the model begins to converge on an F1 score of around 0.6. We can see that the training and test scores have not yet converged, so potentially this model would benefit from more training data. Finally, this model suffers primarily from error due to variance (the CV scores for the test data are more variable than for training data) so it is possible that the model is overfitting.

Regression
----------

Building a learning curve for a regression is straight forward and very similar. In the below example, after loading our data and selecting our target, we explore the learning curve score according to the coefficient of determination or R2 score.

.. code:: python

    from sklearn.linear_model import RidgeCV

    # Load a regression dataset
    data = load_data('energy')

    # Specify features of interest and the target
    targets = ["heating load", "cooling load"]
    features = [col for col in data.columns if col not in targets]

    X = data[features]
    y = data[targets[0]]

    # Create the learning curve visualizer, fit and poof
    viz = LearningCurve(RidgeCV(), train_sizes=sizes, scoring='r2')
    viz.fit(X, y)
    viz.poof()

.. image:: images/learning_curve_regressor.png

This learning curve shows a very high variability and much lower score until about 350 instances. It is clear that this model could benefit from more data because it is converging at a very high score. Potentially, with more data and a larger alpha for regularization, this model would become far less variable in the test data.

Clustering
----------

Learning curves also work for clustering models and can use metrics that specify the shape or organization of clusters such as silhouette scores or density scores. If the membership is known in advance, then rand scores can be used to compare clustering performance as shown below:

.. code:: python

    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    # Create a dataset of blobs
    X, y = make_blobs(n_samples=1000, centers=5)

    viz = LearningCurve(
        KMeans(), train_sizes=sizes, scoring="adjusted_rand_score"
    )

    viz.fit(X, y)
    viz.poof()


.. image:: images/learning_curve_clusterer.png

Unfortunately, with random data these curves are highly variable, but serve to point out some clustering-specific items. First, note the y-axis is very narrow, roughly speaking these curves are converged and actually the clustering algorithm is performing very well. Second, for clustering, convergence for data points is not necessarily a bad thing; in fact we want to ensure as more data is added, the training and cross-validation scores do not diverge.

.. seealso::
    This visualizer is based on the validation curve described in the scikit-learn documentation: `Learning Curves <http://scikit-learn.org/stable/modules/learning_curve.html#learning-curve>`_. The visualizer wraps the `learning_curve <http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve>`_ function and most of the arguments are passed directly to it.


API Reference
-------------

.. automodule:: yellowbrick.model_selection.learning_curve
    :members: LearningCurve
    :undoc-members:
    :show-inheritance:
