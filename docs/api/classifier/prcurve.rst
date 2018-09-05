.. -*- mode: rst -*-

Precision-Recall Curves
=======================

Precision-Recall curves are a metric used to evaluate a classifier's quality,
particularly when classes are very imbalanced. The precision-recall curve
shows the tradeoff between precision, a measure of result relevancy, and
recall, a measure of how many relevant results are returned. A large area
under the curve represents both high recall and precision, the best case
scenario for a classifier, showing a model that returns accurate results
for the majority of classes it selects.



API Reference
-------------

.. automodule:: yellowbrick.classifier.prcurve
    :members: PrecisionRecallCurve
    :undoc-members:
    :show-inheritance:
