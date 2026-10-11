"""
Tests for the learning curve comparison example.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np

from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from yellowbrick.model_selection import LearningCurve

def create_dataset():
    """Create the same nonlinear binary dataset used in the notebook."""
    return make_moons(
        n_samples=1000,
        noise=0.25,
        random_state=42,
    )

def create_cv():
    """Create the cross-validation strategy used in the notebook."""
    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

##########################################################################
# Test Cases
##########################################################################

def test_dataset_size():
    """The example dataset should contain 1,000 observations."""
    x, y = create_dataset()
    assert len(x) == 1000
    assert len(y) == 1000

def test_dataset_has_two_classes():
    """The example dataset should contain exactly two classes."""
    _, y = create_dataset()
    assert set(np.unique(y)) == {0, 1}

def test_logistic_regression_learning_curve():
    """The logistic regression learning curve should fit successfully."""
    x, y = create_dataset()

    visualizer = LearningCurve(
        LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        cv=create_cv(),
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    visualizer.fit(x, y)
    assert visualizer is not None


def test_decision_tree_learning_curve():
    """The decision tree learning curve should fit successfully."""
    x, y = create_dataset()

    visualizer = LearningCurve(
        DecisionTreeClassifier(
            random_state=42,
        ),
        cv=create_cv(),
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
    )

    visualizer.fit(x, y)
    assert visualizer is not None
