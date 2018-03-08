.. -*- mode: rst -*-

Loading a Text Corpus
=====================

As in the previous sections, Yellowbrick has provided a sample dataset to run the following cells. In particular, we are going to use a text corpus wrangled from the `Baleen RSS Corpus <http://baleen.districtdatalabs.com/>`_ to present the following examples. If you haven't already downloaded the data, you can do so by running:

::

    $ python -m yellowbrick.download

Note that this will create a directory called ``data`` in your current working directory that contains subdirectories with the provided datasets.

.. note:: If you've already followed the instructions from :doc:`downloading example datasets <../datasets>`, you don't have to repeat these steps here. Simply check to ensure there is a directory called ``hobbies`` in your data directory.

The following code snippet creates a utility that will load the corpus from disk into a Scikit-Learn Bunch object. This method creates a corpus that is exactly the same as the one found in the `"working with text data" <http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html>`_ example on the Scikit-Learn website, hopefully making the examples easier to use.

.. code:: python

    import os
    from sklearn.datasets.base import Bunch

    def load_corpus(path):
        """
        Loads and wrangles the passed in text corpus by path.
        """

        # Check if the data exists, otherwise download or raise
        if not os.path.exists(path):
            raise ValueError((
                "'{}' dataset has not been downloaded, "
                "use the yellowbrick.download module to fetch datasets"
            ).format(path))

        # Read the directories in the directory as the categories.
        categories = [
            cat for cat in os.listdir(path)
            if os.path.isdir(os.path.join(path, cat))
        ]

        files  = [] # holds the file names relative to the root
        data   = [] # holds the text read from the file
        target = [] # holds the string of the category

        # Load the data from the files in the corpus
        for cat in categories:
            for name in os.listdir(os.path.join(path, cat)):
                files.append(os.path.join(path, cat, name))
                target.append(cat)

                with open(os.path.join(path, cat, name), 'r') as f:
                    data.append(f.read())


        # Return the data bunch for use similar to the newsgroups example
        return Bunch(
            categories=categories,
            files=files,
            data=data,
            target=target,
        )

This is a fairly long ibt of code, so let's walk through it step by step. The data in the corpus directory is stored as follows:

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

Each of the documents in the corpus is stored in a text file labeled with its hash signature in a directory that specifies its label or category. Therefore the first step after checking to make sure the specified path exists is to list all the directories in the ``hobbies`` directory -- this gives us each of our categories, which we will store later in the bunch.

The second step is to create placeholders for holding filenames, text data, and labels. We can then loop through the list of categories, list the files in each category directory, add those files to the files list, add the category name to the target list, then open and read the file to add it to data.

To load the corpus into memory, we can simply use the following snippet:

.. code:: python

    corpus = load_corpus("data/hobbies")

We'll use this snippet in all of the text examples in this section!
