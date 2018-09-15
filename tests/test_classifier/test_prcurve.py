# tests.test_classifier.test_prcurve
# Tests for the Precision-Recall curves visualizer
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Tue Sep 04 16:48:09 2018 -0400
#
# ID: test_prcurve.py [] benjamin@bengfort.com $

"""
Tests for the Precision-Recall curves visualizer
"""

##########################################################################
## Imports
##########################################################################

import sys
import pytest

from yellowbrick.exceptions import *
from yellowbrick.classifier.prcurve import *

from tests.base import VisualTestCase
from .test_rocauc import FakeClassifier

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


##########################################################################
## Assertion Helpers
##########################################################################

LEARNED_FIELDS = (
    'target_type_', 'score_', 'precision_', 'recall_'
)


def assert_not_fitted(oz):
    for field in LEARNED_FIELDS:
        assert not hasattr(oz, field)


def assert_fitted(oz):
    for field in LEARNED_FIELDS:
        assert hasattr(oz, field)



##########################################################################
## PrecisionRecallCurve Tests
##########################################################################

@pytest.mark.usefixtures("binary", "multiclass")
class TestPrecisionRecallCurve(VisualTestCase):
    """
    Test the PrecisionRecallCurve visualizer
    """

    def test_fit_continuous(self):
        """
        Should not allow any target type other than binary or multiclass
        """
        X, y = make_regression()
        with pytest.raises(YellowbrickValueError, match="does not support target type"):
            oz = PrecisionRecallCurve(LinearSVC())
            oz.fit(X, y)

    def test_ensure_fit(self):
        """
        Requires visualizer to be fit
        """
        with pytest.raises(NotFitted, match="cannot wrap an already fitted estimator"):
            oz = PrecisionRecallCurve(RidgeClassifier())
            oz.score(self.binary.X.test, self.binary.y.test)

    def test_binary_probability(self):
        """
        Visual similarity of binary classifier with predict_proba function
        """
        # Create and fit the visualizer
        oz = PrecisionRecallCurve(RandomForestClassifier(random_state=12))
        assert_not_fitted(oz)

        # Fit returns self
        assert oz.fit(self.binary.X.train, self.binary.y.train) is oz

        # Score the visualizer
        s = oz.score(self.binary.X.test, self.binary.y.test)
        assert_fitted(oz)

        # Score should be between 0 and 1
        assert 0.0 <= s <= 1.0

        # Check the binary classification properties
        assert oz.target_type_ == BINARY
        assert isinstance(oz.score_, float)
        assert oz.score_ == s
        assert isinstance(oz.precision_, np.ndarray)
        assert isinstance(oz.recall_, np.ndarray)

        # Compare the images
        oz.finalize()
        tol = 1.5 if sys.platform == 'win32' else 1.0 # fails with RMSE 1.409 on AppVeyor
        self.assert_images_similar(oz, tol=tol)

    def test_binary_probability_decision(self):
        """
        Visual similarity of binary classifier with both predict_proba & decision
        """
        # Create and fit the visualizer
        oz = PrecisionRecallCurve(AdaBoostClassifier(), iso_f1_curves=True)
        assert_not_fitted(oz)

        # Fit returns self
        assert oz.fit(self.binary.X.train, self.binary.y.train) is oz

        # Score the visualizer
        s = oz.score(self.binary.X.test, self.binary.y.test)
        assert_fitted(oz)

        # Score should be between 0 and 1
        assert 0.0 <= s <= 1.0

        # Check the binary classification properties
        assert oz.target_type_ == BINARY
        assert isinstance(oz.score_, float)
        assert oz.score_ == s
        assert isinstance(oz.precision_, np.ndarray)
        assert isinstance(oz.recall_, np.ndarray)

        # Compare the images
        oz.finalize()
        tol = 4.6 if sys.platform == 'win32' else 1.0 # fails with RMSE 4.522 on AppVeyor
        self.assert_images_similar(oz, tol=tol)

    def test_binary_decision(self):
        """
        Visual similarity of binary classifier with a decision function
        """
        # Create and fit the visualizer
        oz = PrecisionRecallCurve(LinearSVC(random_state=232))
        assert_not_fitted(oz)

        # Fit returns self
        assert oz.fit(self.binary.X.train, self.binary.y.train) is oz

        # Score the visualizer
        s = oz.score(self.binary.X.test, self.binary.y.test)
        assert_fitted(oz)

        # Score should be between 0 and 1
        assert 0.0 <= s <= 1.0

        # Check the binary classification properties
        assert oz.target_type_ == BINARY
        assert isinstance(oz.score_, float)
        assert oz.score_ == s
        assert isinstance(oz.precision_, np.ndarray)
        assert isinstance(oz.recall_, np.ndarray)

        # Compare the images
        # NOTE: do not finalize image to ensure tests pass on Travis
        # Fails with 3.083 on Travis-CI (passes on AppVeyor)
        self.assert_images_similar(oz, tol=3.5)

    def test_multiclass_decision(self):
        """
        Visual similarity of multiclass classifier with a decision function
        """
        # Create and fit the visualizer
        oz = PrecisionRecallCurve(RidgeClassifier(random_state=993))
        assert_not_fitted(oz)

        # Fit returns self
        assert oz.fit(self.multiclass.X.train, self.multiclass.y.train) is oz

        # Score the visualizer
        s = oz.score(self.multiclass.X.test, self.multiclass.y.test)
        assert_fitted(oz)

        # Score should be between 0 and 1
        assert 0.0 <= s <= 1.0

        # Check the multiclass classification properties
        assert oz.target_type_ == MULTICLASS
        assert isinstance(oz.score_, dict)
        assert oz.score_[MICRO] == s
        assert isinstance(oz.precision_, dict)
        assert isinstance(oz.recall_, dict)
        assert len(oz.score_) == len(oz.classes_) + 1
        assert len(oz.precision_) == len(oz.classes_) + 1
        assert len(oz.recall_) == len(oz.classes_) + 1

        # Compare the images
        oz.finalize()
        tol = 1.25 if sys.platform == 'win32' else 1.0 # fails with RMSE 1.118 on AppVeyor
        self.assert_images_similar(oz, tol=tol)

    def test_multiclass_probability(self):
        """
        Visual similarity of multiclass classifier with predict_proba function
        """
        # Create and fit the visualizer
        oz = PrecisionRecallCurve(
            GaussianNB(), per_class=True, micro=False, fill_area=False,
            iso_f1_curves=True, ap_score=False
        )
        assert_not_fitted(oz)

        # Fit returns self
        assert oz.fit(self.multiclass.X.train, self.multiclass.y.train) is oz

        # Score the visualizer
        s = oz.score(self.multiclass.X.test, self.multiclass.y.test)
        assert_fitted(oz)

        # Score should be between 0 and 1
        assert 0.0 <= s <= 1.0

        # Check the multiclass classification properties
        assert oz.target_type_ == MULTICLASS
        assert isinstance(oz.score_, dict)
        assert oz.score_[MICRO] == s
        assert isinstance(oz.precision_, dict)
        assert isinstance(oz.recall_, dict)
        assert len(oz.score_) == len(oz.classes_) + 1
        assert len(oz.precision_) == len(oz.classes_) + 1
        assert len(oz.recall_) == len(oz.classes_) + 1

        # Compare the images
        oz.finalize()
        tol = 6.6 if sys.platform == 'win32' else 1.0 # fails with RMSE 6.583 on AppVeyor
        self.assert_images_similar(oz, tol=tol)

    @pytest.mark.filterwarnings("ignore:From version 0.21")
    def test_quick_method(self):
        """
        Test the precision_recall_curve quick method.
        """
        data = load_iris()
        model = DecisionTreeClassifier(random_state=14)

        oz = precision_recall_curve(
            model, data.data, data.target, per_class=True, micro=True,
            fill_area=False,  iso_f1_curves=True, ap_score=False,
            random_state=2)
        assert isinstance(oz, PrecisionRecallCurve)

        tol = 5.8 if sys.platform == 'win32' else 1.0 # fails with RMSE 5.740 on AppVeyor
        self.assert_images_similar(oz, tol=tol)

    def test_no_scoring_function(self):
        """
        Test get y scores with classifiers that have no scoring method
        """
        oz = PrecisionRecallCurve(FakeClassifier())
        with pytest.raises(ModelError, match="requires .* predict_proba or decision_function"):
            oz._get_y_scores(self.binary.X.train)
