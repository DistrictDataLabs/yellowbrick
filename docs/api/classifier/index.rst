.. -*- mode: rst -*-

Classification Visualizers
==========================

Classification models attempt to predict a target in a discrete space, that is assign an instance of dependent variables one or more categories. Classification score visualizers display the differences between classes as well as a number of classifier-specific visual evaluations. We currently have implemented the following classifier evaluations:

-  :doc:`classification_report`: A visual classification report that displays precision, recall, and F1 per-class as a heatmap.
-  :doc:`confusion_matrix`: A heatmap view of the confusion matrix of pairs of classes in multi-class classification.
-  :doc:`rocauc`: Graphs the receiver operating characteristics and area under the curve.
-  :doc:`prcurve`: Plots the precision and recall for different probability thresholds. 
-  :doc:`../target/class_balance`: Visual inspection of the target to show the support of each class to the final estimator.
-  :doc:`class_prediction_error`: An alternative to the confusion matrix that shows both support and the difference between actual and predicted classes.
-  :doc:`threshold`: Shows precision, recall, f1, and queue rate over all thresholds for binary classifiers that use a discrimination probability or score.

Estimator score visualizers wrap scikit-learn estimators and expose the
Estimator API such that they have ``fit()``, ``predict()``, and ``score()``
methods that call the appropriate estimator methods under the hood. Score
visualizers can wrap an estimator and be passed in as the final step in
a ``Pipeline`` or ``VisualPipeline``.

.. code:: python

    # Classifier Evaluation Imports

    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    from yellowbrick.target import ClassBalance
    from yellowbrick.classifier import ROCAUC
    from yellowbrick.classifier import PrecisionRecallCurve
    from yellowbrick.classifier import ClassificationReport
    from yellowbrick.classifier import ClassPredictionError
    from yellowbrick.classifier import DiscriminationThreshold

.. toctree::
   :maxdepth: 2

   classification_report
   confusion_matrix
   rocauc
   prcurve
   class_prediction_error
   threshold
