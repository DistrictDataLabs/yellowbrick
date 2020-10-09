.. -*- mode: rst -*-

Contrib and Third-Party Libraries
=================================

Yellowbrick's primary dependencies are scikit-learn and matplotlib, however the data science landscape in Python is large and there are many opportunties to use Yellowbrick with other machine learning and data analysis frameworks. The ``yellowbrick.contrib`` package contains several methods and mechanisms to support non-scikit-learn machine learning as well as extra tools and experimental visualizers that are outside of core support or are still in development.

.. note:: If you're interested in using a non-scikit-learn estimator with Yellowbrick, please see the :doc:`wrapper` documentation. If the wrapper doesn't work out of the box, we welcome contributions to this module to include other libraries!

The following contrib packages are currently available:

.. toctree::
   :maxdepth: 1

   wrapper
   statsmodels
   boundaries
   scatter
   missing/index
