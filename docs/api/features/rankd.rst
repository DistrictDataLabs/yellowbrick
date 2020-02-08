.. -*- mode: rst -*-

Rank Features
=============

``Rank1D`` and ``Rank2D`` evaluate single features or pairs of features using a variety of metrics that score the features on the scale [-1, 1] or [0, 1] allowing them to be ranked. A similar concept to SPLOMs, the scores are visualized on a lower-left triangle heatmap so that patterns between pairs of features can be easily discerned for downstream analysis.

In this example, we'll use the credit default data set from the UCI Machine Learning repository to rank features. The code below creates our instance matrix and target vector.

=================   ==============================
Visualizers          `Rank1D <https://www.scikit-yb.org/en/latest/api/features/rankd.html#yellowbrick.features.rankd.Rank1D>`_, `Rank2D <https://www.scikit-yb.org/en/latest/api/features/rankd.html#yellowbrick.features.rankd.Rank2D>`_
Quick Methods        `rank2d() <https://www.scikit-yb.org/en/latest/api/features/rankd.html#yellowbrick.features.rankd.rank2d>`_
Models               General Linear Models
Workflow             Feature engineering and model selection
=================   ==============================

Rank 1D
-------

A one-dimensional ranking of features utilizes a ranking algorithm that takes into account only a single feature at a time (e.g. histogram analysis). By default we utilize the Shapiro-Wilk algorithm to assess the normality of the distribution of instances with respect to the feature. A barplot is then drawn showing the relative ranks of each feature.

.. plot::
    :context: close-figs
    :alt: Rank1D on the credit dataset with the Shapiro ranking algorithm

    from yellowbrick.datasets import load_credit
    from yellowbrick.features import Rank1D

    # Load the credit dataset
    X, y = load_credit()

    # Instantiate the 1D visualizer with the Sharpiro ranking algorithm
    visualizer = Rank1D(algorithm='shapiro')

    visualizer.fit(X, y)           # Fit the data to the visualizer
    visualizer.transform(X)        # Transform the data
    visualizer.show()              # Finalize and render the figure


Rank 2D
-------

A two-dimensional ranking of features utilizes a ranking algorithm that takes into account pairs of features at a time (e.g. joint plot analysis). The pairs of features are then ranked by score and visualized using the lower left triangle of a feature co-occurence matrix.

By default, the ``Rank2D`` visualizer utilizes the Pearson correlation score to detect colinear relationships.

.. plot::
    :context: close-figs
    :alt: Rank2D on the credit dataset using Pearson ranking algorithm

    from yellowbrick.datasets import load_credit
    from yellowbrick.features import Rank2D

    # Load the credit dataset
    X, y = load_credit()

    # Instantiate the visualizer with the Pearson ranking algorithm
    visualizer = Rank2D(algorithm='pearson')

    visualizer.fit(X, y)           # Fit the data to the visualizer
    visualizer.transform(X)        # Transform the data
    visualizer.show()              # Finalize and render the figure


Alternatively, we can utilize the covariance ranking algorithm, which attempts to compute the mean value of the product of deviations of variates from their respective means. Covariance loosely attempts to detect a colinear relationship between features. Compare the output from Pearson above to the covariance ranking below.

.. plot::
    :context: close-figs
    :alt: Rank2D on the credit dataset with the covariance algorithm

    from yellowbrick.datasets import load_credit
    from yellowbrick.features import Rank2D

    # Load the credit dataset
    X, y = load_credit()

    # Instantiate the visualizer with the covariance ranking algorithm
    visualizer = Rank2D(algorithm='covariance')

    visualizer.fit(X, y)           # Fit the data to the visualizer
    visualizer.transform(X)        # Transform the data
    visualizer.show()              # Finalize and render the figure

Quick Methods
-------------

Similar functionality as above can be achieved in one line using the associated quick method, ``rank2d``. This method will instantiate and fit a ``Rank2D`` visualizer on the dataset and immediately show it.

.. plot::
    :context: close-figs
    :alt: rank2d quick method on credit dataset with pearson algorithm

    from yellowbrick.datasets import load_credit
    from yellowbrick.features import rank2d

    # Load the credit dataset
    X, _ = load_credit()

    oz = rank2d(X)


API Reference
-------------

.. automodule:: yellowbrick.features.rankd
    :members: Rank1D, Rank2D, rank2d
    :undoc-members:
    :show-inheritance:
