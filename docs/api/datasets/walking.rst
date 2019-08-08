.. -*- mode: rst -*-

Walking
=======

The dataset collects data from an Android smartphone positioned in the chest pocket. Accelerometer Data are collected from 22 participants walking in the wild over a predefined path. The dataset is intended for Activity Recognition research purposes. It provides challenges for identification and authentication of people using motion patterns. Sampling frequency of the accelerometer: DELAY_FASTEST with network connections disabled.

=================   ===========================
Samples total                            149331
Dimensionality                                4
Features                                   real
Targets                               int, 1-22
Task(s)              classification, clustering
=================   ===========================

Description
-----------

In this article, a novel technique for user's authentication and verification using gait as a biometric unobtrusive pattern is proposed. The method is based on a two stages pipeline. First, a general activity recognition classifier is personalized for an specific user using a small sample of her/his walking pattern. As a result, the system is much more selective with respect to the new walking pattern. A second stage verifies whether the user is an authorized one or not. This stage is defined as a one-class classification problem. In order to solve this problem, a four-layer architecture is built around the geometric concept of convex hull. This architecture allows to improve robustness to outliers, modeling non-convex shapes, and to take into account temporal coherence information. Two different scenarios are proposed as validation with two different wearable systems. First, a custom high-performance wearable system is built and used in a free environment. A second dataset is acquired from an Android-based commercial device in a 'wild' scenario with rough terrains, adversarial conditions, crowded places and obstacles. Results on both systems and datasets are very promising, reducing the verification error rates by an order of magnitude with respect to the state-of-the-art technologies.

Citation
--------

Downloaded from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/User+Identification+From+Walking+Activity>`_ on August 23, 2018.

Casale, Pierluigi, Oriol Pujol, and Petia Radeva. "Personalization and user verification in wearable systems using biometric walking patterns." Personal and Ubiquitous Computing 16.5 (2012): 563-580.

Loader
------

.. autofunction:: yellowbrick.datasets.loaders.load_walking
