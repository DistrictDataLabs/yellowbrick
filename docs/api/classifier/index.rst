.. -*- mode: rst -*-

Classification Visualizers
==========================

Classification models attempt to predict a target in a discrete space, that is assign an instance of dependent variables one or more categories. Classification score visualizers display the differences between classes as well as a number of classifier-specific visual evaluations. We currently have implemented four classifier evaluations:

-  :doc:`classification_report`: Presents the classification report of the classifier
   as a heatmap
-  :doc:`confusion_matrix`: Presents the confusion matrix of the classifier
   as a heatmap
-  :doc:`rocauc`: Presents the graph of receiver operating characteristics
   along with area under the curve
-  :doc:`class_balance`: Displays the difference between the class balances and support
-  :doc:`threshold`: Shows the bounds of precision, recall and queue rate after a number of trials.

Estimator score visualizers wrap Scikit-Learn estimators and expose the
Estimator API such that they have fit(), predict(), and score() methods
that call the appropriate estimator methods under the hood. Score
visualizers can wrap an estimator and be passed in as the final step in
a Pipeline or VisualPipeline.

.. code:: python

    # Classifier Evaluation Imports

    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance, ThresholdViz

.. toctree::
   :maxdepth: 2

   classification_report
   confusion_matrix
   rocauc
   class_balance
   threshold
