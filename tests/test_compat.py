# tests.test_compat
# Regression tests for sklearn/numpy/matplotlib compatibility fixes.
# Each test targets a specific breakage that was introduced by a package
# upgrade and verifies the fix continues to work.
#
# Run with: pytest tests/test_compat.py -v

"""
Compatibility regression tests for modernized yellowbrick.

These tests exist specifically to catch regressions in the compatibility
fixes made for scikit-learn 1.6+, NumPy 2.0+, scipy 1.17+, and
matplotlib 3.7+. They are distinct from the normal feature tests and
should never be xfail'd or skipped without a very good reason.
"""

##########################################################################
## Imports
##########################################################################

import numpy as np
import pytest

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, ClusterMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression

from yellowbrick.utils.types import (
    is_classifier,
    is_regressor,
    is_clusterer,
    _get_estimator_type,
)
from yellowbrick.base import ModelVisualizer


##########################################################################
## Helpers
##########################################################################


class MixinOnlyClassifier(ClassifierMixin, BaseEstimator):
    """Classifier that relies solely on MRO for type detection (no _estimator_type)."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class MixinOnlyRegressor(RegressorMixin, BaseEstimator):
    """Regressor that relies solely on MRO for type detection."""

    def fit(self, X, y):
        self.coef_ = np.zeros(X.shape[1])
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MixinOnlyClusterer(ClusterMixin, BaseEstimator):
    """Clusterer that relies solely on MRO for type detection."""

    def fit(self, X, y=None):
        self.labels_ = np.zeros(len(X), dtype=int)
        return self


class MinimalVisualizer(ModelVisualizer):
    """Minimal ModelVisualizer subclass for testing pipeline integration."""

    def fit(self, X, y=None, **kwargs):
        super().fit(X, y, **kwargs)
        return self

    def score(self, X, y, **kwargs):
        return self.estimator.score(X, y)

    def draw(self, **kwargs):
        pass

    def finalize(self, **kwargs):
        pass


##########################################################################
## sklearn 1.6+ estimator type detection (fixes _estimator_type removal)
##########################################################################


class TestEstimatorTypeDetection:
    """
    Tests for _get_estimator_type() and the is_* helpers.

    In sklearn 1.6+, _estimator_type was removed from many estimators.
    Type detection must now fall back to __sklearn_tags__() or Mixin
    subclass checks.
    """

    def test_classifier_via_mixin_class(self):
        """is_classifier works on a class with no _estimator_type attribute."""
        assert is_classifier(MixinOnlyClassifier)

    def test_classifier_via_mixin_instance(self):
        """is_classifier works on an instance with no _estimator_type attribute."""
        assert is_classifier(MixinOnlyClassifier())

    def test_regressor_via_mixin_class(self):
        """is_regressor works on a class with no _estimator_type attribute."""
        assert is_regressor(MixinOnlyRegressor)

    def test_regressor_via_mixin_instance(self):
        """is_regressor works on an instance with no _estimator_type attribute."""
        assert is_regressor(MixinOnlyRegressor())

    def test_clusterer_via_mixin_class(self):
        """is_clusterer works on a class with no _estimator_type attribute."""
        assert is_clusterer(MixinOnlyClusterer)

    def test_clusterer_via_mixin_instance(self):
        """is_clusterer works on an instance with no _estimator_type attribute."""
        assert is_clusterer(MixinOnlyClusterer())

    def test_sklearn_classifier_detected(self):
        """Standard sklearn classifiers are detected correctly."""
        assert is_classifier(LogisticRegression)
        assert is_classifier(LogisticRegression())
        assert is_classifier(GaussianNB)
        assert is_classifier(GaussianNB())

    def test_sklearn_regressor_detected(self):
        """Standard sklearn regressors are detected correctly."""
        assert is_regressor(LinearRegression)
        assert is_regressor(LinearRegression())
        assert is_regressor(Lasso)
        assert is_regressor(Lasso())

    def test_sklearn_clusterer_detected(self):
        """Standard sklearn clusterers are detected correctly."""
        assert is_clusterer(KMeans)
        assert is_clusterer(KMeans())

    def test_cross_type_negative(self):
        """Type checks return False for wrong estimator types."""
        assert not is_classifier(LinearRegression())
        assert not is_regressor(LogisticRegression())
        assert not is_clusterer(LinearRegression())
        assert not is_classifier(KMeans())

    def test_get_estimator_type_returns_none_for_transformer(self):
        """_get_estimator_type returns None for transformers (no type)."""
        assert _get_estimator_type(SimpleImputer()) is None

    def test_get_estimator_type_strings(self):
        """_get_estimator_type returns the correct string for each type."""
        assert _get_estimator_type(LogisticRegression()) == "classifier"
        assert _get_estimator_type(LinearRegression()) == "regressor"
        assert _get_estimator_type(KMeans()) == "clusterer"


##########################################################################
## sklearn 1.8+ Pipeline fitted-state detection (__sklearn_is_fitted__)
##########################################################################


class TestModelVisualizerFittedState:
    """
    Tests for ModelVisualizer.__sklearn_is_fitted__().

    sklearn 1.8 Pipeline.score() calls check_is_fitted(last_step). Without
    __sklearn_is_fitted__(), ModelVisualizer has no trailing-underscore
    attributes of its own, so the pipeline would raise NotFittedError even
    after a successful fit().
    """

    def test_not_fitted_before_fit(self):
        """__sklearn_is_fitted__ returns False before fit is called."""
        viz = MinimalVisualizer(LinearRegression())
        assert not viz.__sklearn_is_fitted__()

    def test_fitted_after_fit(self):
        """__sklearn_is_fitted__ returns True after fit is called."""
        X, y = make_regression(n_samples=50, n_features=4, random_state=0)
        viz = MinimalVisualizer(LinearRegression())
        viz.fit(X, y)
        assert viz.__sklearn_is_fitted__()

    def test_pipeline_score_does_not_raise(self):
        """
        A Pipeline ending with a ModelVisualizer can call .score() without
        raising NotFittedError (regression for sklearn 1.8 compatibility).
        """
        X, y = make_regression(n_samples=100, n_features=4, random_state=0)
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("viz", MinimalVisualizer(LinearRegression())),
            ]
        )

        model.fit(X_train, y_train)
        # This must not raise NotFittedError
        score = model.score(X_test, y_test)
        assert isinstance(score, float)

    def test_pipeline_check_is_fitted_passes(self):
        """sklearn check_is_fitted passes on a Pipeline after fit."""
        from sklearn.utils.validation import check_is_fitted

        X, y = make_regression(n_samples=50, n_features=4, random_state=0)
        model = Pipeline(
            [
                ("viz", MinimalVisualizer(LinearRegression())),
            ]
        )
        model.fit(X, y)
        # Must not raise
        check_is_fitted(model)

    def test_pipeline_check_is_fitted_fails_before_fit(self):
        """sklearn check_is_fitted raises on an unfitted Pipeline."""
        from sklearn.utils.validation import check_is_fitted
        from sklearn.exceptions import NotFittedError

        model = Pipeline(
            [
                ("viz", MinimalVisualizer(LinearRegression())),
            ]
        )
        with pytest.raises(NotFittedError):
            check_is_fitted(model)


##########################################################################
## NumPy 2.0 compatibility
##########################################################################


class TestNumpyCompat:
    """
    Tests for NumPy 2.0 API changes used in yellowbrick source code.
    """

    def test_isin_replaces_in1d(self):
        """np.isin works as the replacement for removed np.in1d."""
        arr = np.array([1, 2, 3, 4, 5])
        assert np.all(np.isin([1, 3], arr))
        assert not np.any(np.isin([6, 7], arr))

    def test_in1d_is_removed(self):
        """np.in1d no longer exists in NumPy 2.0+ (documents the removal)."""
        assert not hasattr(
            np, "in1d"
        ), "np.in1d exists — yellowbrick code should use np.isin instead"

    def test_stack_requires_sequence_not_generator(self):
        """np.stack accepts lists but not bare generators in NumPy 2.0+."""
        arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]

        # List form must work
        result = np.stack(list(arrays))
        assert result.shape == (2, 3)

    def test_bytes_dtype_alias(self):
        """np.bytes_ is the correct alias (np.string_ was removed in 1.24)."""
        assert hasattr(np, "bytes_")
        assert not hasattr(
            np, "string_"
        ), "np.string_ exists — code should use np.bytes_ instead"

    def test_str_dtype_alias(self):
        """np.str_ is the correct alias (np.unicode_ was removed in 1.24)."""
        assert hasattr(np, "str_")
        assert not hasattr(
            np, "unicode_"
        ), "np.unicode_ exists — code should use np.str_ instead"

    def test_percentile_method_kwarg(self):
        """np.percentile accepts 'method' kwarg (was 'interpolation' pre-1.22)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = np.percentile(data, 50, method="nearest")
        assert result in data

    def test_percentile_interpolation_kwarg_removed(self):
        """np.percentile no longer accepts the old 'interpolation' kwarg."""
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError):
            np.percentile(data, 50, interpolation="nearest")


##########################################################################
## matplotlib 3.7+ compatibility
##########################################################################


class TestMatplotlibCompat:
    """
    Tests for matplotlib API changes used in yellowbrick source code.
    """

    def test_stem_no_use_line_collection_kwarg(self):
        """ax.stem() no longer accepts use_line_collection (removed in 3.7)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        data = np.array([0.1, 0.5, 0.3, 0.8])

        # Must not raise TypeError
        ax.stem(data)

        # Passing use_line_collection must raise TypeError
        with pytest.raises(TypeError):
            ax.stem(data, use_line_collection=True)

        plt.close(fig)
