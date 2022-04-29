.. -*- mode: rst -*-

Feature Dropping Curve
=============================

 =================   =====================
 Visualizer           :class:`~yellowbrick.model_selection.dropping_curve.DroppingCurve`
 Quick Method         :func:`~yellowbrick.model_selection.dropping_curve.dropping_curve`
 Models               Classification, Regression
 Workflow             Model Selection
 =================   =====================

A feature dropping curve (FDC) shows the relationship between the score and the number of features used.
This shows how the estimator benefits from additional features of the same type.
For example, how many air quality sensors are needed across a city to accurately predict city-wide pollution levels?

Feature dropping curves helpfully complement :doc:`rfecv` (RFECV).
In the air quality sensor example, RFECV finds which sensors to keep and which sensors can be retired in the specific city.
Feature dropping curves visualize how many sensors a similar-sized city might need to track pollution levels.

Feature dropping curves are common in the field of neural decoding, where they are called `neuron dropping curves <https://dx.doi.org/10.3389%2Ffnsys.2014.00102>`_.
Neural decoding research often quantifies how performance scales with neuron (or electrode) count.
Electrodes do not correspond directly between participants, so we use random feature subsets to simulate what performance we could expect in another participant.


API Reference
-------------

.. automodule:: yellowbrick.model_selection.dropping_curve
    :members: DroppingCurve, dropping_curve
    :undoc-members:
    :show-inheritance:
