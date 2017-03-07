=====
About
=====


About Yellowbrick
-----------------
Yellowbrick is an open source, pure Python project that extends Scikit-Learn with visual analysis and diagnostic tools. The Yellowbrick API also wraps matplotlib to create publication-ready figures and interactive data explorations while still allowing developers fine-grain control of figures. For users, Yellowbrick can help evaluate the performance, stability, and predictive value of machine learning models, and assist in diagnosing problems throughout the machine learning workflow.

The Model Selection Triple
^^^^^^^^^^^^^^^^^^^^^^^^^^
Discussions of machine learning are frequently characterized by a singular focus on model selection. Be it logistic regression, random forests, Bayesian methods, or artificial neural networks, machine learning practitioners are often quick to express their preference. The reason for this is mostly historical. Though modern third-party machine learning libraries have made the deployment of multiple models appear nearly trivial, traditionally the application and tuning of even one of these algorithms required many years of study. As a result, machine learning practitioners tended to have strong preferences for particular (and likely more familiar) models over others.

However, model selection is a bit more nuanced than simply picking the "right" or "wrong" algorithm. In practice, the workflow includes:

  1. selecting and/or engineering the smallest and most predictive feature set
  2. choosing a set of algorithms from a model family, and
  3. tuning the algorithm hyperparameters to optimize performance.

The **model selection triple** was first described in a 2015 SIGMOD_ paper by Kumar et al. In their paper, which concerns the development of next-generation database systems built to anticipate predictive modeling, the authors cogently express that such systems are badly needed due to the highly experimental nature of machine learning in practice. "Model selection," they explain, "is iterative and exploratory because the space of [model selection triples] is usually infinite, and it is generally impossible for analysts to know a priori which [combination] will yield satisfactory accuracy and/or insights."

Recently, much of this workflow has been automated through grid search methods, standardized APIs, and GUI-based applications. In practice, however, human intuition and guidance can more effectively hone in on quality models than exhaustive search. By visualizing the model selection process, data scientists can steer towards final, explainable models and avoid pitfalls and traps.

The Yellowbrick library is a diagnostic visualization platform for machine learning that allows data scientists to steer the model selection process. Yellowbrick extends the Scikit-Learn API with a new core object: the Visualizer. Visualizers allow visual models to be fit and transformed as part of the Scikit-Learn Pipeline process, providing visual diagnostics throughout the transformation of high dimensional data.


Contributing
------------

If you would like to contribute to Yellowbrick, you can do so in the following ways:

- Add issues or bugs to the bug tracker: https://github.com/DistrictDataLabs/yellowbrick/issues
- Work on a card on the dev board: https://waffle.io/DistrictDataLabs/yellowbrick
- Create a pull request in Github: https://github.com/DistrictDataLabs/yellowbrick/pulls

Here's how to get set up with Yellowbrick in development mode (since the project is still under development).

1. Fork and clone the repository. After clicking fork in the upper right corner
for your own copy of yellowbrick to your github account. Clone it in a directory
of your choice.::

    $ git clone https://github.com/[YOURUSERNAME]/yellowbrick
    $ cd yellowbrick

2. Create virtualenv and create the dependencies.::

    $ virtualenv venv
    $ pip install -r requirements.txt

3. Fetch and switch to development.::

    $ git fetch
    $ git checkout develop


Git Conventions
^^^^^^^^^^^^^^^

The Yellowbrick repository is set up in a typical production/release/development cycle_ as described in "A Successful Git Branching Model." A typical workflow is as follows:

1. Select a card from the dev board_, preferably one that is "ready" then move it to "in-progress".

2. Create a branch off of develop called "feature-[feature name]", work and commit into that branch.::

    $ git checkout -b feature-myfeature develop

3. Once you are done working (and everything is tested) merge your feature into develop.::

    $ git checkout develop
    $ git merge --no-ff feature-myfeature
    $ git branch -d feature-myfeature
    $ git push origin develop

4. Repeat. Releases will be routinely pushed into master via release branches, then deployed to the server.

Name Origin
-----------
The Yellowbrick package gets its name from the fictional element in the 1900 children's novel **The Wonderful Wizard of Oz** by American author L. Frank Baum. In the book, the yellow brick road is the path that the protagonist, Dorothy Gale, must travel in order to reach her destination in the Emerald City.

From Wikipedia_:
    "The road is first introduced in the third chapter of The Wonderful Wizard of Oz. The road begins in the heart of the eastern quadrant called Munchkin Country in the Land of Oz. It functions as a guideline that leads all who follow it, to the road's ultimate destination—the imperial capital of Oz called Emerald City that is located in the exact center of the entire continent. In the book, the novel's main protagonist, Dorothy, is forced to search for the road before she can begin her quest to seek the Wizard. This is because the cyclone from Kansas did not release her farmhouse closely near it as it did in the various film adaptations. After the council with the native Munchkins and their dear friend the Good Witch of the North, Dorothy begins looking for it and sees many pathways and roads nearby, (all of which lead in various directions). Thankfully it doesn't take her too long to spot the one paved with bright yellow bricks."


Changelog
---------

Version 0.3.3
^^^^^^^^^^^^^
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
^^^^^^^^^^^^^
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
^^^^^^^^^^^^
Hotfix to solve pip install issues with Yellowbrick.

* Tag: v0.3.1_
* Deployed: Monday, October 10, 2016
* Contributors: Benjamin Bengfort

  Changes:
     - Modified packaging and wheel for Python 2.7 and 3.5 compatibility
     - Modified deployment to PyPI and pip install ability
     - Fixed Travis-CI tests with the backend failures.

Version 0.3
^^^^^^^^^^^
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
^^^^^^^^^^^
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
^^^^^^^^^^^
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


.. _SIGMOD: http://cseweb.ucsd.edu/~arunkk/vision/SIGMODRecord15.pdf
.. _cycle: http://nvie.com/posts/a-successful-git-branching-model/
.. _board: https://waffle.io/districtdatalabs/yellowbrick
.. _Wikipedia: https://en.wikipedia.org/wiki/Yellow_brick_road
.. _v0.3.3: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.3.3
.. _v0.3.2: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.3.2
.. _v0.3.1: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.3.1a2
.. _v0.3: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.3
.. _v0.2: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.2
.. _v0.1: https://github.com/DistrictDataLabs/yellowbrick/releases/tag/v0.1
