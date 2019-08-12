.. -*- mode: rst -*-

Credit
======

This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods.

=================   ===============
Samples total                 30000
Dimensionality                   24
Features                  real, int
Targets                 int, 0 or 1
Task(s)              classification
=================   ===============

Description
-----------

This research aimed at the case of customers' default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods. From the perspective of risk management, the result of predictive accuracy of the estimated probability of default will be more valuable than the binary result of classification - credible or not credible clients. Because the real probability of default is unknown, this study presented the novel "Sorting Smoothing Method" to estimate the real probability of default. With the real probability of default as the response variable (Y), and the predictive probability of default as the independent variable (X), the simple linear regression result (Y = A + BX) shows that the forecasting model produced by artificial neural network has the highest coefficient of determination; its regression intercept (A) is close to zero, and regression coefficient (B) to one. Therefore, among the six data mining techniques, artificial neural network is the only one that can accurately estimate the real probability of default.

Citation
--------

Downloaded from the `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients>`_  on October 13, 2016.

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

Loader
------

.. autofunction:: yellowbrick.datasets.loaders.load_credit
