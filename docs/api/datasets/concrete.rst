.. -*- mode: rst -*-

Concrete
========

Concrete is the most important material in civil engineering. The concrete compressive strength is a highly nonlinear function of age and ingredients.

=================   ===============
Samples total                  1030
Dimensionality                    9
Features                       real
Targets             float, 2.3-82.6
Task(s)                  regression
=================   ===============

Description
-----------

Given are the variable name, variable type, the measurement unit and a brief description. The concrete compressive strength is the regression problem. The order of this listing corresponds to the order of numerals along the rows of the database.

Citation
--------

Downloaded from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength>`_  on October 13, 2016.

Yeh, I-C. "Modeling of strength of high-performance concrete using artificial neural networks." Cement and Concrete research 28.12 (1998): 1797-1808.

Loader
------

.. autofunction:: yellowbrick.datasets.loaders.load_concrete
