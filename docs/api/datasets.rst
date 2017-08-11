.. -*- mode: rst -*-

Example Datasets
================

Yellowbrick hosts several datasets wrangled from the `UCI Machine
Learning Repository <http://archive.ics.uci.edu/ml/>`__ to present the
examples in this section. If you haven't downloaded the data, you can do so by
running:

::

    $ python -m yellowbrick.download

This should create a folder called ``data`` in your current working directory with all of the datasets. You can load a specified dataset with ``pandas.read_csv`` as follows:

.. code:: python

    import pandas as pd

    data = pd.read_csv('data/concrete/concrete.csv')

The following code snippet can be found at the top of the ``examples/examples.ipynb`` notebok in Yellowbrick. Please reference this code when trying to load a specific data set:

.. code:: python

    from yellowbrick.download import download_all

    ## The path to the test data sets
    FIXTURES  = os.path.join(os.getcwd(), "data")

    ## Dataset loading mechanisms
    datasets = {
        "bikeshare": os.path.join(FIXTURES, "bikeshare", "bikeshare.csv"),
        "concrete": os.path.join(FIXTURES, "concrete", "concrete.csv"),
        "credit": os.path.join(FIXTURES, "credit", "credit.csv"),
        "energy": os.path.join(FIXTURES, "energy", "energy.csv"),
        "game": os.path.join(FIXTURES, "game", "game.csv"),
        "mushroom": os.path.join(FIXTURES, "mushroom", "mushroom.csv"),
        "occupancy": os.path.join(FIXTURES, "occupancy", "occupancy.csv"),
    }


    def load_data(name, download=True):
        """
        Loads and wrangles the passed in dataset by name.
        If download is specified, this method will download any missing files.
        """

        # Get the path from the datasets
        path = datasets[name]

        # Check if the data exists, otherwise download or raise
        if not os.path.exists(path):
            if download:
                download_all()
            else:
                raise ValueError((
                    "'{}' dataset has not been downloaded, "
                    "use the download.py module to fetch datasets"
                ).format(name))


        # Return the data frame
        return pd.read_csv(path)

Note that most of the examples currently use one or more of the listed datasets for their examples (unless specifically shown otherwise). Each dataset has a ``README.md`` with detailed information about the data source, attributes, and target. Here is a complete listing of all datasets in Yellowbrick and their associated analytical tasks:

- **bikeshare**: suitable for regression
- **concrete**: suitable for regression
- **credit**: suitable for classification/clustering
- **energy**: suitable for regression
- **game**: suitable for classification
- **hobbies**: suitable for text analysis
- **mushroom**: suitable for classification/clustering
- **occupancy**: suitable for classification
