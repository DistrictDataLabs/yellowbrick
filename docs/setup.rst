===========
Quick Start
===========

Installation
------------

To install the Yellowbrick library, the simplest thing to do is use ``pip`` as follows.::

    $ pip install yellowbrick

Using Yellowbrick
-----------------
The Yellowbrick API is specifically designed to play nicely with Scikit-Learn. Here is an example of a typical workflow sequence with Scikit-Learn and Yellowbrick:

Feature Visualization
^^^^^^^^^^^^^^^^^^^^^
In this example, we see how Rank2D performs pairwise comparisons of each feature in the data set with a specific metric or algorithm, then returns them ranked as a lower left triangle diagram.::

    from yellowbrick.features import Rank2D
    visualizer = Rank2D(features=features, algorithm='covariance')
    visualizer.fit(X, y)                # Fit the data to the visualizer
    visualizer.transform(X)             # Transform the data
    visualizer.poof()                   # Draw/show/poof the data


Model Visualization
^^^^^^^^^^^^^^^^^^^
In this example, we instantiate a Scikit-Learn classifier, and then we use Yellowbrick's ROCAUC class to visualize the tradeoff between the classifier's sensitivity and specificity.::

    from sklearn.svm import LinearSVC
    from yellowbrick.classifier import ROCAUC
    model = LinearSVC()
    model.fit(X,y)
    visualizer = ROCAUC(model)
    visualizer.score(X,y)
    visualizer.poof()


For additional information on getting started with Yellowbrick, check out our :ref:`examples <examples/yellowbrick-examples>`.
