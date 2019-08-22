.. -*- mode: rst -*-
.. yellowbrick documentation master file, created by
    sphinx-quickstart on Tue Jul  5 19:45:43 2016.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

Yellowbrick: Machine Learning Visualization
===========================================

.. image:: images/readme/banner.png

Yellowbrick extends the Scikit-Learn API to make model selection and hyperparameter tuning easier. Under the hood, it's using Matplotlib.

Recommended Learning Path
-------------------------

1. Check out the :doc:`quickstart`, try the :doc:`tutorial`, and check out the :doc:`oneliners`.

2. Use Yellowbrick in your work, referencing the :doc:`api/index` for assistance with specific visualizers and detailed information on optional parameters and customization options.

3. Star us on `GitHub <https://github.com/DistrictDataLabs/yellowbrick/>`_ and follow us on `Twitter (@scikit_yb) <https://twitter.com/scikit_yb>`_ so that you'll hear about new visualizers as soon as they're added.

Contributing
------------

Interested in contributing to Yellowbrick? Yellowbrick is a welcoming, inclusive project and we would love to have you.
We follow the `Python Software Foundation Code of Conduct <http://www.python.org/psf/codeofconduct/>`_.

No matter your level of technical skill, you can be helpful. We appreciate bug reports, user testing, feature requests, bug fixes, product enhancements, and documentation improvements.

Check out the :doc:`contributing/index` guide!

If you've signed up to do user testing, head over to the :doc:`evaluation`.

Please consider joining the `Google Groups Listserv <https://groups.google.com/forum/#!forum/yellowbrick>`_ listserve so you can respond to questions.

Thank you for your contributions!

Concepts & API
--------------

Visualizers
-----------
The primary goal of Yellowbrick is to create a sensical API similar to Scikit-Learn.

Visualizers are the core objects in Yellowbrick.
They are similar to transformers in Scikit-Learn.
Visualizers can wrap a model estimator - similar to how the "ModelCV" (e.g. `RidgeCV <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html>`_, `LassoCV <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html>`_) methods work.


Some of our most popular visualizers include:

Feature Visualization
~~~~~~~~~~~~~~~~~~~~~

- :doc:`api/features/rankd`: pairwise ranking of features to detect relationships
- :doc:`api/features/pcoords`: horizontal visualization of instances
- :doc:`Radial Visualization <api/features/radviz>`: separation of instances around a circular plot
- :doc:`api/features/pca`: projection of instances based on principal components
- :doc:`api/features/manifold`: high dimensional visualization with manifold learning
- :doc:`Joint Plots <api/features/jointplot>`: direct data visualization with feature selection

Classification Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`api/classifier/class_prediction_error`: shows error and support in classification
- :doc:`api/classifier/classification_report`: visual representation of precision, recall, and F1
- :doc:`ROC/AUC Curves <api/classifier/rocauc>`: receiver operator characteristics and area under the curve
- :doc:`api/classifier/prcurve`: precision vs recall for different probability thresholds
- :doc:`Confusion Matrices <api/classifier/confusion_matrix>`: visual description of class decision making
- :doc:`Discrimination Threshold <api/classifier/threshold>`: find a threshold that best separates binary classes

Regression Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`api/regressor/peplot`: find model breakdowns along the domain of the target
- :doc:`api/regressor/residuals`: show the difference in residuals of training and test data
- :doc:`api/regressor/alphas`: show how the choice of alpha influences regularization
- :doc:`api/regressor/influence`: show the influence of instances on linear regression

Clustering Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`K-Elbow Plot <api/cluster/elbow>`: select k using the elbow method and various metrics
- :doc:`Silhouette Plot <api/cluster/silhouette>`: select k by visualizing silhouette coefficient values
- :doc:`api/cluster/icdm`: show relative distance and size/importance of clusters

Model Selection Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  :doc:`api/model_selection/validation_curve`: tune a model with respect to a single hyperparameter
-  :doc:`api/model_selection/learning_curve`: show if a model might benefit from more data or less complexity
-  :doc:`api/model_selection/importances`: rank features by importance or linear coefficients for a specific model
-  :doc:`api/model_selection/rfecv`: find the best subset of features based on importance

Target Visualization
~~~~~~~~~~~~~~~~~~~~

- :doc:`api/target/binning`: generate a histogram with vertical lines showing the recommended value point to bin the data into evenly distributed bins
- :doc:`api/target/class_balance`: see how the distribution of classes affects the model
- :doc:`api/target/feature_correlation`: display the correlation between features and dependent variables

Text Visualization
~~~~~~~~~~~~~~~~~~

- :doc:`Term Frequency <api/text/freqdist>`: visualize the frequency distribution of terms in the corpus
- :doc:`api/text/tsne`: use stochastic neighbor embedding to project documents
- :doc:`api/text/dispersion`: visualize how key terms are dispersed throughout a corpus
- :doc:`api/text/umap_vis`: plot similar documents closer together to discover clusters
- :doc:`api/text/postag`: plot the counts of different parts-of-speech throughout a tagged corpus

... and more! Visualizers are being added all the time. Check the examples (or even the `develop branch <https://github.com/DistrictDataLabs/yellowbrick/tree/develop>`_). Feel free to contribute your ideas for new Visualizers!

Getting Help
------------

Can't get someting to work? Here are places you can find help.

1. The docs (you're here!).
2. `Stack Overflow <http://stackoverflow.com/questions/tagged/yellowbrick>`_. If you ask a question, please tag it with "yellowbrick".
3. The Yellowbrick `Google Groups Listserv <https://groups.google.com/forum/#!forum/yellowbrick>`_.
4. You can also Tweet or direct message us on Twitter `@scikit_yb <https://twitter.com/scikit_yb>`_.


Find a Bug?
-----------

Check if there's already an open `issue <https://github.com/DistrictDataLabs/yellowbrick/issues/>`_ on the topic. If needed, file an `issue <https://github.com/DistrictDataLabs/yellowbrick/issues/>`_.


Open Source
-----------

The Yellowbrick `license <https://github.com/DistrictDataLabs/yellowbrick/blob/master/LICENSE.txt>`_ is an open source `Apache 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_ license. Yellowbrick enjoys a very active developer community; please consider :doc:`contributing/index`!

Yellowbrick is hosted on `GitHub <https://github.com/DistrictDataLabs/yellowbrick/>`_. The `issues <https://github.com/DistrictDataLabs/yellowbrick/issues/>`_ and `pull requests <https://github.com/DistrictDataLabs/yellowbrick/pulls>`_ are tracked there.


Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   quickstart
   tutorial
   api/index
   oneliners
   contributing/index
   matplotlib
   teaching
   gallery
   about
   faq
   evaluation
   code_of_conduct
   changelog
   governance/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
