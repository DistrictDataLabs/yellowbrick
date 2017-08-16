.. -*- mode: rst -*-

Changelog
=========

Version 0.5
-----------

* Tag: v0.5_
* Deployed: Wednesday, August 9, 2017
* Contributors: Benjamin Bengfort, Rebecca Bilbro, Nathan Danielsen, Carlo Morales, Jim Stearns, Phillip Schafer, Jason Keung

Changes:
    - Added VisualTestCase.
    - New PCADecomposition Visualizer, which decomposes high dimensional data into two or three dimensions so that each instance can be plotted in a scatter plot.
    - New and improved ROCAUC Visualizer, which now supports multiclass classification.
    - Prototype Decision Boundary Visualizer, which is a bivariate data visualization algorithm that plots the decision boundaries of each class.
    - Added Rank1D Visualizer, which is a one dimensional ranking of features that utilizes the Shapiro-Wilks ranking that takes into account only a single feature at a time (e.g. histogram analysis).
    - Improved Prediction Error Plot with identity line, shared limits, and r squared.
    - Updated FreqDist Visualizer to make word features a hyperparameter.
    - Added normalization and scaling to Parallel Coordinates.
    - Added Learning Curve Visualizer, which displays a learning curve based on the number of samples versus the training and cross validation scores to show how a model learns and improves with experience.
    - Added data downloader module to the yellowbrick library.
    - Complete overhaul of the yellowbrick documentation; categories of methods are located in separate pages to make it easier to read and contribute to the documentation.
    - Added a new color palette inspired by `ANN-generated colors <http://lewisandquark.tumblr.com/>`_

Bug Fixes:
   - Repairs to PCA, RadViz, FreqDist unit tests
   - Repair to matplotlib version check in JointPlot Visualizer

.. _v0.5: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.5

Hotfix 0.4.2
------------

Update to the deployment docs and package on both Anaconda and PyPI.

* Tag: v0.4.2_
* Deployed: Monday, May 22, 2017
* Contributors: Benjamin Bengfort, Jason Keung

.. _v0.4.2: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.4.2


Version 0.4.1
-------------
This release is an intermediate version bump in anticipation of the PyCon 2017 sprints.

The primary goals of this version were to (1) update the Yellowbrick dependencies (2) enhance the Yellowbrick documentation to help orient new users and contributors, and (3) make several small additions and upgrades (e.g. pulling the Yellowbrick utils into a standalone module).

We have updated the Scikit-Learn and SciPy dependencies from version 0.17.1 or later to 0.18 or later. This primarily entails moving from ``from sklearn.cross_validation import train_test_split`` to ``from sklearn.model_selection import train_test_split``.

The updates to the documentation include new Quickstart and Installation guides as well as updates to the Contributors documentation, which is modeled on the Scikit-Learn contributing documentation.

This version also included upgrades to the KMeans visualizer, which now supports not only ``silhouette_score`` but also ``distortion_score`` and ``calinski_harabaz_score``. The ``distortion_score`` computes the mean distortion of all samples as the sum of the squared distances between each observation and its closest centroid. This is the metric that K-Means attempts to minimize as it is fitting the model. The ``calinski_harabaz_score`` is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.

Finally, this release includes a prototype of the ``VisualPipeline``, which extends Scikit-Learn's ``Pipeline`` class, allowing multiple Visualizers to be chained or sequenced together.

* Tag: v0.4.1_
* Deployed: Monday, May 22, 2017
* Contributors: Benjamin Bengfort, Rebecca Bilbro, Nathan Danielsen

Changes:
   - Score and model visualizers now wrap estimators as proxies so that all methods on the estimator can be directly accessed from the visualizer
   - Updated Scikit-learn dependency from >=0.17.1  to >=0.18
   - Replaced ``sklearn.cross_validation`` with ``model_selection``
   - Updated SciPy dependency from >=0.17.1 to >=0.18
   - ScoreVisualizer now subclasses ModelVisualizer; towards allowing both fitted and unfitted models passed to Visualizers
   - Added CI tests for Python 3.6 compatibility
   - Added new quickstart guide and install instructions
   - Updates to the contributors documentation
   - Added ``distortion_score`` and ``calinski_harabaz_score`` computations and visualizations to KMeans visualizer.
   - Replaced the ``self.ax`` property on all of the individual ``draw`` methods with a new property on the ``Visualizer`` class that ensures all visualizers automatically have axes.
   - Refactored the utils module into a package
   - Continuing to update the docstrings to conform to Sphinx
   - Added a prototype visual pipeline class that extends the Scikit-learn pipeline class to ensure that visualizers get called correctly.

Bug Fixes:
   - Fixed title bug in Rank2D FeatureVisualizer

.. _v0.4.1: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.4.1


Version 0.4
-----------
This release is the culmination of the Spring 2017 DDL Research Labs that focused on developing Yellowbrick as a community effort guided by a sprint/agile workflow. We added several more visualizers, did a lot of user testing and bug fixes, updated the documentation, and generally discovered how best to make Yellowbrick a friendly project to contribute to.

Notable in this release is the inclusion of two new feature visualizers that use few, simple dimensions to visualize features against the target. The ``JointPlotVisualizer`` graphs a scatter plot of two dimensions in the data set and plots a best fit line across it. The ``ScatterVisualizer`` also uses two features, but also colors the graph by the target variable, adding a third dimension to the visualization.

This release also adds support for clustering visualizations, namely the elbow method for selecting K, ``KElbowVisualizer`` and a visualization of cluster size and density using the ``SilhouetteVisualizer``. The release also adds support for regularization analysis using the ``AlphaSelection`` visualizer. Both the text and classification modules were also improved with the inclusion of the ``PosTagVisualizer`` and the ``ConfusionMatrix`` visualizer respectively.

This release also added an Anaconda repository and distribution so that users can ``conda install`` yellowbrick. Even more notable, we got yellowbrick stickers! We've also updated the documentation to make it more friendly and a bit more visual; fixing the API rendering errors. All-in-all, this was a big release with a lot of contributions and we thank everyone that participated in the lab!

* Tag: v0.4_
* Deployed: Thursday, May 4, 2017
* Contributors: Benjamin Bengfort, Rebecca Bilbro, Nathan Danielsen, Matt Andersen, Prema Roman, Neal Humphrey, Jason Keung, Bala Venkatesan, Paul Witt, Morgan Mendis, Tuuli Morril

Changes:
   - Part of speech tags visualizer -- ``PosTagVisualizer``.
   - Alpha selection visualizer for regularized regression -- ``AlphaSelection``
   - Confusion Matrix Visualizer -- ``ConfusionMatrix``
   - Elbow method for selecting K vis -- ``KElbowVisualizer``
   - Silhouette score cluster visualization -- ``SilhouetteVisualizer``
   - Joint plot visualizer with best fit -- ``JointPlotVisualizer``
   - Scatter visualization of features -- ``ScatterVisualizer``
   - Added three more example datasets: mushroom, game, and bike share
   - Contributor's documentation and style guide
   - Maintainers listing and contacts
   - Light/Dark background color selection utility
   - Structured array detection utility
   - Updated classification report to use colormesh
   - Added anacondas packaging and distribution
   - Refactoring of the regression, cluster, and classification modules
   - Image based testing methodology
   - Docstrings updated to a uniform style and rendering
   - Submission of several more user studies

Version 0.3.3
-------------
Intermediate sprint to demonstrate prototype implementations of text visualizers for NLP models. Primary contributions were the ``FreqDistVisualizer`` and the ``TSNEVisualizer``.

The ``TSNEVisualizer`` displays a projection of a vectorized corpus in two dimensions using TSNE, a nonlinear dimensionality reduction method that is particularly well suited to embedding in two or three dimensions for visualization as a scatter plot. TSNE is widely used in text analysis to show clusters or groups of documents or utterances and their relative proximities.

The ``FreqDistVisualizer`` implements frequency distribution plot that tells us the frequency of each vocabulary item in the text. In general, it could count any kind of observable event. It is a distribution because it tells us how the total number of word tokens in the text are distributed across the vocabulary items.

* Tag: v0.3.3_
* Deployed: Wednesday, February 22, 2017
* Contributors: Rebecca Bilbro, Benjamin Bengfort

Changes:
   - TSNEVisualizer for 2D projections of vectorized documents
   - FreqDistVisualizer for token frequency of text in a corpus
   - Added the user testing evaluation to the documentation
   - Created scikit-yb.org and host documentation there with RFD
   - Created a sample corpus and text examples notebook
   - Created a base class for text, ``TextVisualizer``
   - Model selection tutorial using Mushroom Dataset
   - Created a text examples notebook but have not added to documentation.


Version 0.3.2
-------------
Hardened the Yellowbrick API to elevate the idea of a Visualizer to a first principle. This included reconciling shifts in the development of the preliminary versions to the new API, formalizing Visualizer methods like `draw()` and `finalize()`, and adding utilities that revolve around Scikit-Learn. To that end we also performed administrative tasks like refreshing the documentation and preparing the repository for more and varied open source contributions.

* Tag: v0.3.2_
* Deployed: Friday, January 20, 2017
* Contributors: Benjamin Bengfort, Rebecca Bilbro

Changes:
   - Converted Mkdocs documentation to Sphinx documentation
   - Updated docstrings for all Visualizers and functions
   - Created a DataVisualizer base class for dataset visualization
   - Single call functions for simple visualizer interaction
   - Added yellowbrick specific color sequences and palettes and env handling
   - More robust examples with downloader from DDL host
   - Better axes handling in visualizer, matplotlib/sklearn integration
   - Added a finalize method to complete drawing before render
   - Improved testing on real data sets from examples
   - Bugfix: score visualizer renders in notebook but not in Python scripts.
   - Bugfix: tests updated to support new API

Hotfix 0.3.1
-------------
Hotfix to solve pip install issues with Yellowbrick.

* Tag: v0.3.1_
* Deployed: Monday, October 10, 2016
* Contributors: Benjamin Bengfort

  Changes:
     - Modified packaging and wheel for Python 2.7 and 3.5 compatibility
     - Modified deployment to PyPI and pip install ability
     - Fixed Travis-CI tests with the backend failures.

Version 0.3
-----------
This release marks a major change from the previous MVP releases as Yellowbrick moves towards direct integration with Scikit-Learn for visual diagnostics and steering of machine learning and could therefore be considered the first alpha release of the library. To that end we have created a Visualizer model which extends sklearn.base.BaseEstimator and can be used directly in the ML Pipeline. There are a number of visualizers that can be used throughout the model selection process, including for feature analysis, model selection, and hyperparameter tuning.

In this release specifically we focused on visualizers in the data space for feature analysis and visualizers in the model space for scoring and evaluating models. Future releases will extend these base classes and add more functionality.

* Tag: v0.3_
* Deployed: Sunday, October 9, 2016
* Contributors: Benjamin Bengfort, Rebecca Bilbro, Marius van Niekerk

  Enhancements:
   - Created an API for visualization with machine learning: Visualizers that are BaseEstimators.
   - Created a class hierarchy for Visualizers throughout the ML process particularly feature analysis and model evaluation
   - Visualizer interface is draw method which can be called multiple times on data or model spaces and a poof method to finalize the figure and display or save to disk.
   - ScoreVisualizers wrap Scikit-Learn estimators and implement fit and predict (pass-throughs to the estimator) and also score which calls draw in order to visually score the estimator. If the estimator isn't appropriate for the scoring method an exception is raised.
   - ROCAUC is a ScoreVisualizer that plots the receiver operating characteristic curve and displays the area under the curve score.
   - ClassificationReport is a ScoreVisualizer that renders the confusion matrix of a classifier as a heatmap.
   - PredictionError is a ScoreVisualizer that plots the actual vs. predicted values and the 45 degree accuracy line for regressors.
   - ResidualPlot is a ScoreVisualizer that plots the residuals (y - yhat) across the actual values (y) with the zero accuracy line for both train and test sets.
   - ClassBalance is a ScoreVisualizer that displays the support for each class as a bar plot.
   - FeatureVisualizers are Scikit-Learn Transformers that implement fit and transform and operate on the data space, calling draw to display instances.
   - ParallelCoordinates plots instances with class across each feature dimension as line segments across a horizontal space.
   - RadViz plots instances with class in a circular space where each feature dimension is an arc around the circumference and points are plotted relative to the weight of the feature.
   - Rank2D plots pairwise scores of features as a heatmap in the space [-1, 1] to show relative importance of features. Currently implemented ranking functions are Pearson correlation and covariance.
   - Coordinated and added palettes in the bgrmyck space and implemented a version of the Seaborn set_palette and set_color_codes functions as well as the ColorPalette object and other matplotlib.rc modifications.
   - Inherited Seaborn's notebook context and whitegrid axes style but make them the default, don't allow user to modify (if they'd like to, they'll have to import Seaborn). This gives Yellowbrick a consistent look and feel without giving too much work to the user and prepares us for Matplotlib 2.0.
   - Jupyter Notebook with Examples of all Visualizers and usage.

  Bug Fixes:
   - Fixed Travis-CI test failures with matplotlib.use('Agg').
   - Fixed broken link to Quickstart on README
   - Refactor of the original API to the Scikit-Learn Visualizer API

Version 0.2
-----------
Intermediate steps towards a complete API for visualization. Preparatory stages for Scikit-Learn visual pipelines.

* Tag: v0.2_
* Deployed: Sunday, September 4, 2016
* Contributors: Benjamin Bengfort, Rebecca Bilbro, Patrick O'Melveny, Ellen Lowy, Laura Lorenz

  Changes:
   - Continued attempts to fix the Travis-CI Scipy install failure (broken tests)
   - Utility function: get the name of the model
   - Specified a class based API and the basic interface (render, draw, fit, predict, score)
   - Added more documentation, converted to Sphinx, autodoc, docstrings for viz methods, and a quickstart
   - How to contribute documentation, repo images etc.
   - Prediction error plot for regressors (mvp)
   - Residuals plot for regressors (mvp)
   - Basic style settings a la seaborn
   - ROC/AUC plot for classifiers (mvp)
   - Best fit functions for "select best", linear, quadratic
   - Several Jupyter notebooks for examples and demonstrations



Version 0.1
-----------
Created the yellowbrick library MVP with two primary operations: a classification report heat map and a ROC/AUC curve model analysis for classifiers. This is the base package deployment for continuing yellowbrick development.

* Tag: v0.1_
* Deployed: Wednesday, May 18, 2016
* Contributors: Benjamin Bengfort, Rebecca Bilbro

  Changes:
   - Created the Anscombe quartet visualization example
   - Added DDL specific color maps and a stub for more style handling
   - Created crplot which visualizes the confusion matrix of a classifier
   - Created rocplot_compare which compares two classifiers using ROC/AUC metrics
   - Stub tests/stub documentation


.. _v0.4: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.4
.. _v0.3.3: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.3.3
.. _v0.3.2: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.3.2
.. _v0.3.1: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.3.1a2
.. _v0.3: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.3
.. _v0.2: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.2
.. _v0.1: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.1
