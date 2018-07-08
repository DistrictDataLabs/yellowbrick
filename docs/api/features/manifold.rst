.. -*- mode: rst -*-

Manifold Visualization
======================

The ``Manifold`` visualizer provides high dimensional visualization using
`manifold learning`_
to embed instances described by many dimensions into 2, thus allowing the
creation of a scatter plot that shows latent structures in data. Unlike
decomposition methods such as PCA and SVD, manifolds generally use
nearest-neighbors approaches to embedding, allowing them to capture non-linear
structures that would be otherwise lost. The projections that are produced
can then be analyzed for noise or separability to determine if it is possible
to create a decision space in the data.

.. image:: images/concrete_tsne_manifold.png

The ``Manifold`` visualizer allows access to all currently available
scikit-learn manifold implementations by specifying the manifold as a string to the visualizer. The currently implemented default manifolds are as follows:

==============  ============================================================
Manifold        Description
--------------  ------------------------------------------------------------
``"lle"``       `Locally Linear Embedding`_ (LLE) uses many local linear
                decompositions to preserve globally non-linear structures.
``"ltsa"``      `LTSA LLE`_: local tangent space alignment is similar to LLE
                in that it uses locality to preserve neighborhood distances.
``"hessian"``   `Hessian LLE`_ an LLE regularization method that applies a
                hessian-based quadratic form at each neighborhood
``"modified"``  `Modified LLE`_ applies a regularization parameter to LLE.
``"isomap"``    `Isomap`_ seeks a lower dimensional embedding that maintains
                geometric distances between each instance.
``"mds"``       `MDS`_: multi-dimensional scaling uses similarity to plot
                points that are near to each other close in the embedding.
``"spectral"``  `Spectral Embedding`_ a discrete approximation of the low
                dimensional manifold using a graph representation.
``"tsne"``      `t-SNE`_: converts the similarity of points into probabilities
                then uses those probabilities to create an embedding.
==============  ============================================================

Each manifold algorithm produces a different embedding and takes advantage of
different properties of the underlying data. Generally speaking, it requires
multiple attempts on new data to determine the manifold that works best for
the structures latent in your data. Note however, that different manifold
algorithms have different time, complexity, and resource requirements.

Manifolds can be used on many types of problems, and the color used in the
scatter plot can describe the target instance. In an unsupervised or
clustering problem, a single color is used to show structure and overlap. In
a classification problem discrete colors are used for each class. In a
regression problem, a color map can be used to describe points as a heat map
of their regression values.

Discrete Target
---------------

In a classification or clustering problem, the instances can be described by
discrete labels - the classes or categories in the supervised problem, or the
clusters they belong to in the unsupervised version. The manifold visualizes
this by assigning a color to each label and showing the labels in a legend.

.. code:: python

    # Load the classification data set
    data = load_data('occupancy')

    # Specify the features of interest
    features = [
        "temperature", "relative humidity", "light", "C02", "humidity"
    ]

    # Extract the data from the data frame.
    X = data[features]
    y = data.occupancy

.. code:: python

    from yellowbrick.features.manifold import Manifold

    visualizer = Manifold(manifold='tsne', target='discrete')
    visualizer.fit_transform(X,y)
    visualizer.poof()


.. image:: images/occupancy_tsne_manifold.png

The visualization also displays the amount of time it takes to generate the
embedding; as you can see, this can take a long time even for relatively
small datasets. One tip is scale your data using the ``StandardScalar``;
another is to sample your instances (e.g. using ``train_test_split`` to
preserve class stratification) or to filter features to decrease sparsity in
the dataset.

One common mechanism is to use `SelectKBest` to select the features that have
a statistical correlation with the target dataset. For example, we can use
the ``f_classif`` score to find the 3 best features in our occupancy dataset.

.. code:: python

    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    model = Pipeline([
        ("selectk", SelectKBest(k=3, score_func=f_classif)),
        ("viz", Manifold(manifold='isomap', target='discrete')),
    ])

    X, y = load_occupancy_data()
    model.fit(X, y)
    model.named_steps['viz'].poof()

.. image:: images/occupancy_select_k_best_isomap_manifold.png

Continuous Target
-----------------

For a regression target or to specify color as a heat-map of continuous
values, specify ``target='continuous'``. Note that by default the param
``target='auto'`` is set, which determines if the target is discrete or
continuous by counting the number of unique values in ``y``.

.. code:: python

    # Specify the features of interest
    feature_names = [
        'cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age'
    ]
    target_name = 'strength'

    # Get the X and y data from the DataFrame
    X = data[feature_names]
    y = data[target_name]

.. code:: python

    visualizer = Manifold(manifold='isomap', target='continuous')
    visualizer.fit_transform(X,y)
    visualizer.poof()

.. image:: images/concrete_isomap_manifold.png

API Reference
-------------

.. automodule:: yellowbrick.features.manifold
    :members: Manifold
    :undoc-members:
    :show-inheritance:


.. _`manifold learning`: http://scikit-learn.org/stable/modules/manifold.html
.. _`manifold comparisons`: http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
.. _`Locally Linear Embedding`: http://scikit-learn.org/stable/modules/manifold.html#locally-linear-embedding
.. _`LTSA LLE`: http://scikit-learn.org/stable/modules/manifold.html#local-tangent-space-alignment
.. _`Hessian LLE`: http://scikit-learn.org/stable/modules/manifold.html#hessian-eigenmapping>
.. _`Modified LLE`: http://scikit-learn.org/stable/modules/manifold.html#modified-locally-linear-embedding
.. _`Isomap`: http://scikit-learn.org/stable/modules/manifold.html#isomap
.. _`MDS`: http://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds
.. _`Spectral Embedding`: http://scikit-learn.org/stable/modules/manifold.html#spectral-embedding
.. _`t-SNE`: http://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne
