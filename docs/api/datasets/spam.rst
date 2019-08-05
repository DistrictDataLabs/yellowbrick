.. -*- mode: rst -*-

Spam
====

Classifying Email as Spam or Non-Spam.

=================   =====================================
Samples total                                        4601
Dimensionality                                         57
Features                                    real, integer
Targets                 int: {1 for spam, 0 for not spam}
Task(s)                                    classification
=================   =====================================


Description
-----------

The "spam" concept is diverse: advertisements for products/web sites, make
money fast schemes, chain letters, pornography...

Our collection of spam e-mails came from our postmaster and individuals who had filed spam. Our collection of non-spam e-mails came from filed work and personal e-mails, and hence the word 'george' and the area code '650' are indicators of non-spam. These are useful when constructing a personalized spam filter. One would either have to blind such non-spam indicators or get a very wide collection of non-spam to generate a general purpose spam filter.

Determine whether a given email is spam or not.

~7% misclassification error. False positives (marking good mail as spam) are very undesirable.If we insist on zero false positives in the training/testing set, 20-25% of the spam passed through the filter.

Citation
--------
Downloaded from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/spambase>`_ on March 23, 2018.

Cranor, Lorrie Faith, and Brian A. LaMacchia. "Spam!." Communications of the ACM 41.8 (1998): 74-83.

Loader
------

.. autofunction:: yellowbrick.datasets.loaders.load_spam
