.. -*- mode: rst -*-

Occupancy
=========

Experimental data used for binary classification (room occupancy) from Temperature, Humidity, Light and CO2. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.

=================   ===============
Classes                           2
Samples per class        imbalanced
Samples total                 20560
Dimensionality                    5
Features             real, positive
=================   ===============

Citation
--------

Downloaded from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+>`_ on October 13, 2016.

Candanedo, Luis M., and Véronique Feldheim. "Accurate occupancy detection of an office room from light, temperature, humidity and CO 2 measurements using statistical learning models." Energy and Buildings 112 (2016): 28-39.

.. autofunction:: yellowbrick.datasets.loaders.load_occupancy
