Rank2D
======

Rank2D visualizers evaluate single features or pairs of features using a
variety of metrics that score the features on the scale [-1, 1] or [0,
1] allowing them to be ranked. A similar concept to SPLOMs, the scores
are visualized on a lower-left triangle heatmap so that patterns between
pairs of features can be easily discerned for downstream analysis.

.. code:: python

    # Load the classification data set
    data = load_data('credit')

    # Specify the features of interest
    features = [
            'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
            'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
            'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', 'jun_pay',
            'jul_pay', 'aug_pay', 'sep_pay',
        ]

    # Extract the numpy arrays from the data frame
    X = data[features].as_matrix()
    y = data.default.as_matrix()

.. code:: python

    # Instantiate the visualizer with the Covariance ranking algorithm
    visualizer = Rank2D(features=features, algorithm='covariance')

    visualizer.fit(X, y)                # Fit the data to the visualizer
    visualizer.transform(X)             # Transform the data
    visualizer.poof()    # Draw/show/poof the data


.. image:: images/rank2d_covariance.png


.. code:: python

    # Instantiate the visualizer with the Pearson ranking algorithm
    visualizer = Rank2D(features=features, algorithm='pearson')

    visualizer.fit(X, y)                # Fit the data to the visualizer
    visualizer.transform(X)             # Transform the data
    visualizer.poof()                   # Draw/show/poof the data



.. image:: images/rank2d_pearson.png


API Reference
-------------

.. automodule:: yellowbrick.features.rankd
    :members: Rank2D
    :undoc-members:
    :show-inheritance:
