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


Each dataset has a ``README.md`` with detailed information about the data source, attributes, and target as well as other metadata. To get access to the metadata or to more precisely control your data access you can return the dataset directly from the loader as follows:

.. code:: python

    dataset = load_bikeshare(return_dataset=True)
    print(dataset.README)

    df = dataset.to_dataframe()
    df.head()


Datasets
--------

Unless otherwise specified, most of the documentation examples currently use one or more of the listed datasets. Here is a complete listing of all datasets in Yellowbrick and the analytical tasks with which they are most commonly associated:

.. Below is a custom ToC, please add new datasets both to the list with a link and
   the file containing the dataset information to the toctree directive below.

- :doc:`bikeshare`: suitable for regression
- :doc:`concrete`: suitable for regression
- :doc:`credit`: suitable for classification/clustering
- :doc:`energy`: suitable for regression
- :doc:`game`: suitable for multi-class classification
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


Yellowbrick has included these datasets in our package for demonstration purposes only. The datasets have been repackaged with the permission of the authors or in accordance with the terms of use of the source material. If you use a Yellowbrick wrangled dataset, please be sure to cite the original author.

API Reference
-------------

By default, the dataset loaders return a features table, ``X``, and a target vector ``y`` when called. If the user has Pandas installed, the data types will be a ``pd.DataFrame`` and ``pd.Series`` respectively, otherwise the data will be returned as numpy arrays. This functionality ensures that the primary use of the datasets, to follow along with the documentation examples, is as simple as possible. However, advanced users may note that there does exist an underlying object with advanced functionality that can be accessed as follows:

.. code:: python

    dataset = load_occupancy(return_dataset=True)


There are two basic types of dataset, the ``Dataset`` which is used for :ref:`tabular data <tabular-data>` loaded from a CSV and the ``Corpus``, used to load :ref:`text corpora <text-corpora>` from disk. Both types of dataset give access to a readme file, a citation in BibTex format, json metadata that describe the fields and target, and different data types associated with the underlying datasset. Both objects are also responsible for locating the dataset on disk and downloading it safely if it doesn't exist yet. For more on how Yellowbrick downloads and stores data, please see :ref:`local-storage`.

.. _tabular-data:

Tabular Data
~~~~~~~~~~~~

Most example datasets are returned as tabular data structures loaded either from a .csv file (using Pandas) or from dtype encoded .npz file to ensure correct numpy arrays are returned. The ``Dataset`` object loads the data from these stored files, preferring to use Pandas if it is installed. It then uses metadata to slice the DataFrame into a feature matrix and target array. Using the dataset directly provides extra functionality, and can be retrieved as follows:

.. code:: python

    from yellowbrick.datasets import load_concrete
    dataset = load_concrete(return_dataset=True)

For example if you wish to get the raw data frame you can do so as follows:

.. code:: python

    df = dataset.to_dataframe()
    df.head()

There may be additional columns in the DataFrame that were part of the original dataset but were excluded from the featureset. For example, the :doc:`energy dataset <energy>` contains two targets, the heating and the cooling load, but only the heating load is returned by default. The api documentation that follows describes in details the metadata properties and other functionality associated with the ``Dataset``:

.. autoclass:: yellowbrick.datasets.base.Dataset
    :show-inheritance:
    :members:
    :inherited-members:

.. _text-corpora:

Text Corpora
~~~~~~~~~~~~

Yellowbrick supports many text-specific machine learning visualizations in the :doc:`yellowbrick.text <../text/index>` module. To facilitate these examples and show an end-to-end visual diagnostics workflow that includes text preprocessing, Yellowbrick supports a ``Corpus`` dataset loader that provides access to raw text data from individual documents. Most notably used with the :doc:`hobbies corpus <hobbies>`, a collection of blog posts from different topics that can be used for text classification tasks.

A text corpus is composed of individual documents that are stored on disk in a directory structure that also identifies document relationships. The file name of each document is a unique file ID (e.g. the MD5 hash of its contents). For example, the hobbies corpus is structured as follows:

::

    data/hobbies
    ├── README.md
    └── books
    |   ├── 56d62a53c1808113ffb87f1f.txt
    |   └── 5745a9c7c180810be6efd70b.txt
    └── cinema
    |   ├── 56d629b5c1808113ffb87d8f.txt
    |   └── 57408e5fc180810be6e574c8.txt
    └── cooking
    |   ├── 56d62b25c1808113ffb8813b.txt
    |   └── 573f0728c180810be6e2575c.txt
    └── gaming
    |   ├── 56d62654c1808113ffb87938.txt
    |   └── 574585d7c180810be6ef7ffc.txt
    └── sports
        ├── 56d62adec1808113ffb88054.txt
        └── 56d70f17c180810560aec345.txt

Unlike the ``Dataset``, corpus dataset loaders do not return ``X`` and ``y`` specially prepared for machine learning. Instead, these loaders return a ``Corpus`` object, which can be used to get a more detailed view of the dataset. For example, to list the unique categories in the corpus, you would access the ``labels`` property as follows:

.. code:: python

    from yellowbrick.datasets import load_hobbies

    corpus = load_hobbies()
    corpus.labels

Addtionally, you can access the list of the absolute paths of each file, which allows you to read individual documents or to use scikit-learn utilties that read the documents streaming one at a time rather than loading them into memory all at once.

.. code:: python

    with open(corpus.files[8], 'r') as f:
        print(f.read())

To get the raw text data and target labels, use the ``data`` and ``target`` properties.

.. code:: python

    X, y = corpus.data, corpus.target

For more details on the other metadata properties associated with the ``Corpus``, please refer to the API reference below. For more detail on text analytics and machine learning with scikit-learn, please refer to `"Working with Text Data" <https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html>`_ in the scikit-learn documentation.

.. autoclass:: yellowbrick.datasets.base.Corpus
    :show-inheritance:
    :members:
    :inherited-members:

.. _local-storage:

Local Storage
~~~~~~~~~~~~~

Yellowbrick datasets are stored in a compressed format in the cloud to ensure that the install process is as streamlined and lightweight as possible. When you request a dataset via the loader module, Yellowbrick checks to see if it has been downloaded already, and if not, it downloads it to your local disk.

By default the dataset is stored, uncompressed, in the ``site-packages`` folder of your Python installation alongside the Yellowbrick code. This means that if you install Yellowbrick in multiple virtual environments, the datasets will be downloaded multiple times in each environment.

To cleanup downloaded datasets, you may use the download module as a command line tool. Note, however, that this will only cleanup the datasets in the yellowbrick package that is on the ``$PYTHON_PATH`` of the environment you're currently in.

.. code::

    $ python -m yellowbrick.download --cleanup --no-download

Alternatively, because the data is stored in the same directory as the code, you can simply ``pip uninstall yellowbrick`` to cleanup the data.

A better option may be to use a single dataset directory across all virtual environments. To specify this directory, you must set the ``$YELLOWBRICK_DATA`` environment variable, usually by adding it to your bash profile so it is exported every time you open a terminal window. This will ensure that you have only downloaded the data once.

.. code::

    $ export YELLOWBRICK_DATA="~/.yellowbrick"
    $ python -m yellowbrick.download -f
    $ ls $YELLOWBRICK_DATA

To identify the location that the Yellowbrick datasets are stored for your installation of Python/Yellowbrick, you can use the ``get_data_home`` function:

.. autofunction:: yellowbrick.datasets.path.get_data_home
