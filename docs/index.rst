.. -*- mode: rst -*-
.. yellowbrick documentation master file, created by
    sphinx-quickstart on Tue Jul  5 19:45:43 2016.
    You can adapt this file completely to your liking, but it should at least
    contain the root `toctree` directive.

Yellowbrick: Machine Learning Visualization
===========================================

.. image:: images/visualizers.png

Yellowbrick is a suite of visual diagnostic tools called "Visualizers" that extend the Scikit-Learn API to allow human steering of the model selection process. In a nutshell, Yellowbrick combines scikit-learn with matplotlib in the best tradition of the scikit-learn documentation, but to produce visualizations for *your* models! For more on Yellowbrick, please see the :doc:`about`.

If you're new to Yellowbrick, checkout the :doc:`quickstart` or skip ahead to the :doc:`tutorial`. Yellowbrick is a rich library with many Visualizers being added on a regular basis. For details on specific Visualizers and extended usage head over to the :doc:`api/index`. Interested in contributing to Yellowbrick? Checkout the :ref:`contributing guide <contributing>` . If you've signed up to do user testing, head over to the :doc:`evaluation` (and thank you!).

Visualizers
-----------

Visualizers are estimators (objects that learn from data) whose primary objective is to create visualizations that allow insight into the model selection process. In Scikit-Learn terms, they can be similar to transformers when visualizing the data space or wrap an model estimator similar to how the "ModelCV" (e.g. `RidgeCV <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html>`_, `LassoCV <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html>`_) methods work. The primary goal of Yellowbrick is to create a sensical API similar to Scikit-Learn. Some of our most popular visualizers include:

Feature Visualization
~~~~~~~~~~~~~~~~~~~~~

- :doc:`api/features/rankd`: pairwise ranking of features to detect relationships
- :doc:`api/features/pcoords`: horizontal visualization of instances
- :doc:`Radial Visualization <api/features/radviz>`: separation of instances around a circular plot
- :doc:`api/features/pca`: projection of instances based on principal components
- :doc:`api/features/manifold`: high dimensional visualization with manifold learning
- :doc:`api/features/importances`: rank features by importance or linear coefficients for a specific model
- :doc:`api/features/rfecv`: find the best subset of features based on importance
- :doc:`Joint Plots <api/features/jointplot>`: direct data visualization with feature selection

Classification Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`api/classifier/class_prediction_error`: shows error and support in classification
- :doc:`api/classifier/classification_report`: visual representation of precision, recall, and F1
- :doc:`ROC/AUC Curves <api/classifier/rocauc>`: receiver operator characteristics and area under the curve
-  :doc:`api/classifier/prcurve`: precision vs recall for different probability thresholds
- :doc:`Confusion Matrices <api/classifier/confusion_matrix>`: visual description of class decision making
- :doc:`Discrimination Threshold <api/classifier/threshold>`: find a threshold that best separates binary classes

Regression Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`api/regressor/peplot`: find model breakdowns along the domain of the target
- :doc:`api/regressor/residuals`: show the difference in residuals of training and test data
- :doc:`api/regressor/alphas`: show how the choice of alpha influences regularization

Clustering Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`K-Elbow Plot <api/cluster/elbow>`: select k using the elbow method and various metrics
- :doc:`Silhouette Plot <api/cluster/silhouette>`: select k by visualizing silhouette coefficient values
- :doc:`api/cluster/icdm`: show relative distance and size/importance of clusters

Model Selection Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  :doc:`api/model_selection/validation_curve`: tune a model with respect to a single hyperparameter
-  :doc:`api/model_selection/learning_curve`: show if a model might benefit from more data or less complexity

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

... and more! Visualizers are being added all the time; be sure to check the examples (or even the `develop branch <https://github.com/DistrictDataLabs/yellowbrick/tree/develop>`_) and feel free to contribute your ideas for new Visualizers!

Getting Help
------------

Yellowbrick is a welcoming, inclusive project in the tradition of matplotlib and scikit-learn. Similar to those projects, we follow the `Python Software Foundation Code of Conduct <http://www.python.org/psf/codeofconduct/>`_. Please don't hesitate to reach out to us for help or if you have any contributions or bugs to report!

The primary way to ask for help with Yellowbrick is to post on our `Google Groups Listserv <https://groups.google.com/forum/#!forum/yellowbrick>`_. This is an email list/forum that members of the community can join and respond to each other; you should be able to receive the quickest response here. Please also consider joining the group so you can respond to questions! You can also ask questions on `Stack Overflow <http://stackoverflow.com/questions/tagged/yellowbrick>`_ and tag them with "yellowbrick". Or you can add issues on GitHub. You can also tweet or direct message us on Twitter `@scikit_yb <https://twitter.com/scikit_yb>`_.

Open Source
-----------

The Yellowbrick `license <https://github.com/DistrictDataLabs/yellowbrick/blob/master/LICENSE.txt>`_ is an open source `Apache 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_ license. Yellowbrick enjoys a very active developer community; please consider joining them and `contributing <https://github.com/DistrictDataLabs/yellowbrick/blob/develop/CONTRIBUTING.md>`_!

Yellowbrick is hosted on `GitHub <https://github.com/DistrictDataLabs/yellowbrick/>`_. The `issues <https://github.com/DistrictDataLabs/yellowbrick/issues/>`_ and `pull requests <https://github.com/DistrictDataLabs/yellowbrick/pulls>`_ are tracked there.


Table of Contents
-----------------

The following is a complete listing of the Yellowbrick documentation for this version of the library:

.. toctree::
   :maxdepth: 2

   quickstart
   tutorial
   api/index
   evaluation
   contributing/index
   matplotlib
   teaching
   gallery
   about
   faq
   code_of_conduct
   changelog
   governance

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
