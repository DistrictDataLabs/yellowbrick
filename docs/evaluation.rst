.. -*- mode: rst -*-

User Testing Instructions
=========================

We are looking for people to help us Alpha test the Yellowbrick project!
Helping is simple: simply create a notebook that applies the concepts in
this *Getting Started* guide to a small-to-medium size dataset of your
choice. Run through the examples with the dataset, and try to change
options and customize as much as possible. After you've exercised the
code with your examples, respond to our `alpha testing
survey <https://goo.gl/forms/naoPUMFa1xNcafY83>`__!

Step One: Questionaire
~~~~~~~~~~~~~~~~~~~~~~
Please open the quesionaire, in order to familiarize yourself with the
feedback that we are looking to receive. We are very interested in
identifying any bugs in Yellowbrick. Please include al cells in your
jupyter notebook that produce errors so that we may reproduce the
problem.


Step Two: Dataset
~~~~~~~~~~~~~~~~~

Select a multivariate dataset of your own; the more (e.g. different)
datasets that we can run through Yellowbrick, the more likely we'll
discover edge cases and exceptions! Note that your dataset must be
well-suited to modeling with Scikit-Learn. In particular we recommend
you choose a dataset whose target is suited to the following supervised
learning tasks:

-  `Regression <https://en.wikipedia.org/wiki/Regression_analysis>`__
   (target is a continuous variable)
-  `Classification <https://en.wikipedia.org/wiki/Classification_in_machine_learning>`__
   (target is a discrete variable)

There are datasets that are well suited to both types of analysis;
either way you can use the testing methodology from this notebook for
either type of task (or both). In order to find a dataset, we recommend
you try the following places:

-  `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml/>`__
-  `MLData.org <http://mldata.org/>`__
-  `Awesome Public
   Datasets <https://github.com/caesar0301/awesome-public-datasets>`__

You're more than welcome to choose a dataset of your own, but we do ask
that you make at least the notebook containing your testing results
publicly available for us to review. If the data is also public (or
you're willing to share it with the primary contributors) that will help
us figure out bugs and required features much more easily!

Step Three: Notebook
~~~~~~~~~~~~~~~~~~~~

Create a notebook in a GitHub repository. We suggest the following:

1. Fork the Yellowbrick repository
2. Under the ``examples`` directory, create a directory named with your
   GitHub username
3. Create a notebook named ``testing``, i.e. examples/USERNAME/testing.ipynb

Alternatively, you could just send us a notebook via Gist or your own
repository. However, if you fork Yellowbrick, you can initiate a pull
request to have your example added to our gallery!

Step Four: Model with Yellowbrick and Scikit-Learn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the following to the notebook:

-  A title in markdown
-  A description of the dataset and where it was obtained
-  A section that loads the data into a Pandas dataframe or NumPy matrix

Then conduct the following modeling activities:

-  Feature analysis using Scikit-Learn and Yellowbrick
-  Estimator fitting using Scikit-Learn and Yellowbrick

You can follow along with our ``examples`` directory (check out
`examples.ipynb <https://github.com/DistrictDataLabs/yellowbrick/blob/master/examples/examples.ipynb>`__)
or even create your own custom visualizers! The goal is that you create
an end-to-end model from data loading to estimator(s) with visualizers
along the way.

**IMPORTANT**: please make sure you record all errors that you get and
any tracebacks you receive for step three!

Step Five: Feedback
~~~~~~~~~~~~~~~~~~~

Finally, submit feedback via the Google Form we have created:

https://goo.gl/forms/naoPUMFa1xNcafY83

This form is allowing us to aggregate multiple submissions and bugs so
that we can coordinate the creation and management of issues. If you are
the first to report a bug or feature request, we will make sure you're
notified (we'll tag you using your Github username) about the created
issue!

Step Six: Thanks!
~~~~~~~~~~~~~~~~~

Thank you for helping us make Yellowbrick better! We'd love to see pull
requests for features you think would be extend the library. We'll also
be doing a user study that we would love for you to participate in. Stay
tuned for more great things from Yellowbrick!
