.. -*- mode: rst -*-

Occupancy
=========

Experimental data used for binary classification (room occupancy) from Temperature, Humidity, Light and CO2. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.

=================   ==========================================
Samples total                                            20560
Dimensionality                                               6
Features                                        real, positive
Targets              int: {1 for occupied, 0 for not occupied}
Task(s)                                         classification
Samples per class                                   imbalanced
=================   ==========================================

Description
-----------

Three data sets are submitted, for training and testing. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute. For the journal publication, the processing R scripts can be found on `GitHub <https://github.com/LuisM78/Occupancy-detection-data>`_.

Citation
--------

Downloaded from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+>`_ on October 13, 2016.

Candanedo, Luis M., and Véronique Feldheim. "Accurate occupancy detection of an office room from light, temperature, humidity and CO 2 measurements using statistical learning models." Energy and Buildings 112 (2016): 28-39.

.. autofunction:: yellowbrick.datasets.loaders.load_occupancy
