.. -*- mode: rst -*-

Class Balance
=============

One of the biggest challenges for classification models is an imbalance of classes in the training data. Severe class imbalances may be masked by relatively good F1 and accuracy scores -- the classifier is simply guessing the majority class and not making any evaluation on the underrepresented class.

There are several techniques for dealing with class imbalance such as stratified sampling, down sampling the majority class, weighting, etc. But before these actions can be taken, it is important to understand what the class balance is in the training data. The ``ClassBalance`` visualizer supports this by creating a bar chart of the *support* for each class, that is the frequency of the classes' representation in the dataset.

=================   ==============================
Visualizer           :class:`~yellowbrick.target.class_balance.ClassBalance`
Quick Method         :func:`~yellowbrick.target.class_balance.class_balance`
Models               Classification
Workflow             Feature analysis, Target analysis, Model selection
=================   ==============================


.. plot::
    :context: close-figs
    :alt: ClassBalance Visualizer on the game dataset

    from yellowbrick.datasets import load_game
    from yellowbrick.target import ClassBalance

    # Load the classification dataset
    X, y = load_game()

    # Instantiate the visualizer
    visualizer = ClassBalance(labels=["draw", "loss", "win"])

    visualizer.fit(y)        # Fit the data to the visualizer
    visualizer.show()        # Finalize and render the figure

The resulting figure allows us to diagnose the severity of the balance issue. In this figure we can see that the ``"win"`` class dominates the other two classes. One potential solution might be to create a binary classifier: ``"win"`` vs ``"not win"`` and combining the ``"loss"`` and ``"draw"`` classes into one class.

.. warning::
    The ``ClassBalance`` visualizer interface has changed in version 0.9, a classification model is no longer required to instantiate the visualizer, it can operate on data only. Additionally, the signature of the fit method has changed from ``fit(X, y=None)`` to ``fit(y_train, y_test=None)``, passing in ``X`` is no longer required.

If a class imbalance must be maintained during evaluation (e.g. the event being classified is actually as rare as the frequency implies) then *stratified sampling* should be used to create train and test splits. This ensures that the test data has roughly the same proportion of classes as the training data. While scikit-learn does this by default in ``train_test_split`` and other ``cv`` methods, it can be useful to compare the support of each class in both splits.

The ``ClassBalance`` visualizer has a "compare" mode, where the train and test data can be passed to ``fit()``, creating a side-by-side bar chart instead of a single bar chart as follows:

.. plot::
    :context: close-figs
    :alt: ClassBalance Visualizer on the occupancy dataset

    from sklearn.model_selection import TimeSeriesSplit

    from yellowbrick.datasets import load_occupancy
    from yellowbrick.target import ClassBalance

    # Load the classification dataset
    X, y = load_occupancy()

    # Create the training and test data
    tscv = TimeSeriesSplit()
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Instantiate the visualizer
    visualizer = ClassBalance(labels=["unoccupied", "occupied"])

    visualizer.fit(y_train, y_test)        # Fit the data to the visualizer
    visualizer.show()                      # Finalize and render the figure


This visualization allows us to do a quick check to ensure that the proportion of each class is roughly similar in both splits. This visualization should be a first stop particularly when evaluation metrics are highly variable across different splits.

.. note:: This example uses ``TimeSeriesSplit`` to split the data into the training and test sets. For more information on this cross-validation method, please refer to the scikit-learn `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html>`_.


Quick Method
------------

The same functionalities above can be achieved with the associated quick method `class_balance`. This method will build the ``ClassBalance`` object with the associated arguments, fit it, then (optionally) immediately show it.

.. plot::
    :context: close-figs
    :alt: class_balance on the game dataset

    from yellowbrick.datasets import load_game
    from yellowbrick.target import class_balance

    # Load the dataset
    X, y = load_game()

    # Use the quick method and immediately show the figure
    class_balance(y)


API Reference
-------------

.. automodule:: yellowbrick.target.class_balance
    :members: ClassBalance, class_balance
    :undoc-members:
    :show-inheritance:
