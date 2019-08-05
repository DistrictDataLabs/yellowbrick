.. -*- mode: rst -*-

Example Datasets
================

Yellowbrick hosts several datasets wrangled from the `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml/>`__ to present the examples used throughout this documentation. These datasets are hosted in our CDN and must be downloaded for use. Typically, when a user calls one of the data loader functions, e.g. ``load_bikeshare()`` the data is automatically downloaded if it's not already on the user's computer. However, for development and testing, or if you know you will be working without internet access, it might be easier to simply download all the data at once.

The data downloader script can be run as follows:

::

    $ python -m yellowbrick.download

This will download all of the data to the ``fixtures`` directory inside of the Yellowbrick site packages. You can specify the location of the download either as an argument to the downloader script (use ``--help`` for more details) or by setting the ``$YELLOWBRICK_DATA`` environment variable. This is the preferred mechanism because this will also influence how data is loaded in Yellowbrick.

.. NOTE:: Developers who have downloaded data from Yellowbrick versions earlier than v1.0 may experience some problems with the older data format. If this occurs, you can clear out your data cache by running ``python -m yellowbrick.download --cleanup``. This will remove old datasets and download the new ones. You can also use the ``--no-download`` flag to simply clear the cache without re-downloading data. Users who are having difficulty with datasets can also use this or they can uninstall and reinstall Yellowbrick using ``pip``.

Once you have downloaded the example datasets, you can load and use them as follows:

.. code:: python

    from yellowbrick.datasets import load_bikeshare

    X, y = load_bikeshare() # returns features and targets for the bikeshare dataset


Unless otherwise specified, most of the examples currently use one or more of the listed datasets. Each dataset has a ``README.md`` with detailed information about the data source, attributes, and target. Here is a complete listing of all datasets in Yellowbrick and the analytical tasks with which they are most commonly associated:

.. Below is a custom ToC, please add new datasets both to the list with a link and
   the file containing the dataset information to the toctree directive below.

- :doc:`bikeshare`: suitable for regression
- :doc:`concrete`: suitable for regression
- :doc:`credit`: suitable for classification/clustering
- :doc:`energy`: suitable for regression
- :doc:`game`: suitable for classification
- :doc:`hobbies`: suitable for text analysis/classification
- :doc:`mushroom`: suitable for classification/clustering
- :doc:`occupancy`: suitable for classification
- :doc:`spam`: suitable for binary classification
- :doc:`walking`: suitable for time series analysis/clustering
- :doc:`nfl`: suitable for clustering

.. toctree::
   :hidden:

   bikeshare
   concrete
   credit
   energy
   game
   hobbies
   mushroom
   occupancy
   spam
   walking
   nfl


API Reference
-------------

.. automodule:: yellowbrick.datasets.base
    :members: BaseDataset, Dataset, Corpus
    :undoc-members:
    :show-inheritance:

.. autofunction:: yellowbrick.datasets.path.get_data_home
