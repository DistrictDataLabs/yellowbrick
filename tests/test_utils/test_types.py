# tests.test_utils.test_types
# Tests for type checking utilities and validation
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Fri May 19 10:58:32 2017 -0700
#
# ID: test_types.py [79cd8cf] benjamin@bengfort.com $

"""
Tests for type checking utilities and validation.

Generally if there is a problem with a type checking utility, the offending
object should be imported then added to the correct bucket under the import
statement (e.g. REGRESSORS). The pytest parametrize decorator uses these
groups to generate tests, so this will automatically cause the test to run on
that class.
"""

##########################################################################
## Imports
##########################################################################

import pytest
import inspect

try:
    import pandas as pd
except:
    pd = None

# Yellowbrick Utilities
from yellowbrick.utils.types import *
from yellowbrick.base import Visualizer, ScoreVisualizer, ModelVisualizer

# Import Regressors
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

REGRESSORS = [
    SVR,
    DecisionTreeRegressor,
    MLPRegressor,
    LinearRegression,
    RandomForestRegressor,
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
]

# Import Classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB

CLASSIFIERS = [
    SVC,
    DecisionTreeClassifier,
    MLPClassifier,
    LogisticRegression,
    RandomForestClassifier,
    GradientBoostingClassifier,
    MultinomialNB,
    GaussianNB,
]

# Import Clusterers
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AffinityPropagation, Birch

CLUSTERERS = [KMeans, MiniBatchKMeans, AffinityPropagation, Birch]

# Import Decompositions
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

DECOMPOSITIONS = [PCA, TruncatedSVD]

# Import Transformers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

TRANSFORMERS = [
    DictVectorizer,
    QuantileTransformer,
    StandardScaler,
    SimpleImputer,
    TfidfVectorizer,
]

# Import Pipeline Utilities
from sklearn.pipeline import Pipeline, FeatureUnion


PIPELINES = [Pipeline, FeatureUnion]

# Import GridSearch Utilities
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

SEARCH = [GridSearchCV, RandomizedSearchCV]


# Other Groups
MODELS = REGRESSORS + CLASSIFIERS + CLUSTERERS
ESTIMATORS = MODELS + DECOMPOSITIONS + TRANSFORMERS


# Get the name of the object to label test cases
def obj_name(obj):
    if inspect.isclass(obj):
        return obj.__name__
    return obj.__class__.__name__


##########################################################################
## Model type checking test cases
##########################################################################


class TestModelTypeChecking(object):
    """
    Test model type checking utilities
    """

    ##////////////////////////////////////////////////////////////////////
    ## is_estimator testing
    ##////////////////////////////////////////////////////////////////////

    def test_estimator_alias(self):
        """
        Assert isestimator aliases is_estimator
        """
        assert isestimator is is_estimator

    @pytest.mark.parametrize("model", ESTIMATORS, ids=obj_name)
    def test_is_estimator(self, model):
        """
        Test that is_estimator works for instances and classes
        """
        assert inspect.isclass(model)
        assert is_estimator(model)

        obj = model()
        assert is_estimator(obj)

    @pytest.mark.parametrize(
        "cls", [list, dict, tuple, set, str, bool, int, float], ids=obj_name
    )
    def test_not_is_estimator(self, cls):
        """
        Assert Python objects are not estimators
        """
        assert inspect.isclass(cls)
        assert not is_estimator(cls)

        obj = cls()
        assert not is_estimator(obj)

    def test_is_estimator_pipeline(self):
        """
        Test that is_estimator works for pipelines
        """
        assert is_estimator(Pipeline)
        assert is_estimator(FeatureUnion)

        model = Pipeline([("reduce_dim", PCA()), ("linreg", LinearRegression())])

        assert is_estimator(model)

    def test_is_estimator_search(self):
        """
        Test that is_estimator works for search
        """
        assert is_estimator(GridSearchCV)
        assert is_estimator(RandomizedSearchCV)

        model = GridSearchCV(SVR(), {"kernel": ["linear", "rbf"]})
        assert is_estimator(model)

    @pytest.mark.parametrize(
        "viz,params",
        [
            (Visualizer, {}),
            (ScoreVisualizer, {"estimator": LinearRegression()}),
            (ModelVisualizer, {"estimator": LogisticRegression()}),
        ],
        ids=["Visualizer", "ScoreVisualizer", "ModelVisualizer"],
    )
    def test_is_estimator_visualizer(self, viz, params):
        """
        Test that is_estimator works for Visualizers
        """
        assert inspect.isclass(viz)
        assert is_estimator(viz)

        obj = viz(**params)
        assert is_estimator(obj)

    ##////////////////////////////////////////////////////////////////////
    ## is_regressor testing
    ##////////////////////////////////////////////////////////////////////

    def test_regressor_alias(self):
        """
        Assert isregressor aliases is_regressor
        """
        assert isregressor is is_regressor

    @pytest.mark.parametrize("model", REGRESSORS, ids=obj_name)
    def test_is_regressor(self, model):
        """
        Test that is_regressor works for instances and classes
        """
        assert inspect.isclass(model)
        assert is_regressor(model)

        obj = model()
        assert is_regressor(obj)

    @pytest.mark.parametrize(
        "model", CLASSIFIERS + CLUSTERERS + TRANSFORMERS + DECOMPOSITIONS, ids=obj_name
    )
    def test_not_is_regressor(self, model):
        """
        Test that is_regressor does not match non-regressor estimators
        """
        assert inspect.isclass(model)
        assert not is_regressor(model)

        obj = model()
        assert not is_regressor(obj)

    def test_is_regressor_pipeline(self):
        """
        Test that is_regressor works for pipelines
        """
        assert not is_regressor(Pipeline)
        assert not is_regressor(FeatureUnion)

        model = Pipeline([("reduce_dim", PCA()), ("linreg", LinearRegression())])

        assert is_regressor(model)

    @pytest.mark.xfail(reason="grid search has no _estimator_type it seems")
    def test_is_regressor_search(self):
        """
        Test that is_regressor works for search
        """
        assert is_regressor(GridSearchCV)
        assert is_regressor(RandomizedSearchCV)

        model = GridSearchCV(SVR(), {"kernel": ["linear", "rbf"]})
        assert is_regressor(model)

    @pytest.mark.parametrize(
        "viz,params",
        [
            (ScoreVisualizer, {"estimator": LinearRegression()}),
            (ModelVisualizer, {"estimator": Ridge()}),
        ],
        ids=["ScoreVisualizer", "ModelVisualizer"],
    )
    def test_is_regressor_visualizer(self, viz, params):
        """
        Test that is_regressor works on visualizers
        """
        assert inspect.isclass(viz)
        assert not is_regressor(viz)

        obj = viz(**params)
        assert is_regressor(obj)

    ##////////////////////////////////////////////////////////////////////
    ## is_classifier testing
    ##////////////////////////////////////////////////////////////////////

    def test_classifier_alias(self):
        """
        Assert isclassifier aliases is_classifier
        """
        assert isclassifier is is_classifier

    @pytest.mark.parametrize("model", CLASSIFIERS, ids=obj_name)
    def test_is_classifier(self, model):
        """
        Test that is_classifier works for instances and classes
        """
        assert inspect.isclass(model)
        assert is_classifier(model)

        obj = model()
        assert is_classifier(obj)

    @pytest.mark.parametrize(
        "model", REGRESSORS + CLUSTERERS + TRANSFORMERS + DECOMPOSITIONS, ids=obj_name
    )
    def test_not_is_classifier(self, model):
        """
        Test that is_classifier does not match non-classifier estimators
        """
        assert inspect.isclass(model)
        assert not is_classifier(model)

        obj = model()
        assert not is_classifier(obj)

    def test_classifier_pipeline(self):
        """
        Test that is_classifier works for pipelines
        """
        assert not is_classifier(Pipeline)
        assert not is_classifier(FeatureUnion)

        model = Pipeline([("reduce_dim", PCA()), ("linreg", LogisticRegression())])

        assert is_classifier(model)

    @pytest.mark.xfail(reason="grid search has no _estimator_type it seems")
    def test_is_classifier_search(self):
        """
        Test that is_classifier works for search
        """
        assert is_classifier(GridSearchCV)
        assert is_classifier(RandomizedSearchCV)

        model = GridSearchCV(SVC(), {"kernel": ["linear", "rbf"]})
        assert is_classifier(model)

    @pytest.mark.parametrize(
        "viz,params",
        [
            (ScoreVisualizer, {"estimator": MultinomialNB()}),
            (ModelVisualizer, {"estimator": MLPClassifier()}),
        ],
        ids=["ScoreVisualizer", "ModelVisualizer"],
    )
    def test_is_classifier_visualizer(self, viz, params):
        """
        Test that is_classifier works on visualizers
        """
        assert inspect.isclass(viz)
        assert not is_classifier(viz)

        obj = viz(**params)
        assert is_classifier(obj)

    ##////////////////////////////////////////////////////////////////////
    ## is_clusterer testing
    ##////////////////////////////////////////////////////////////////////

    def test_clusterer_alias(self):
        """
        Assert isclusterer aliases is_clusterer
        """
        assert isclusterer is is_clusterer

    @pytest.mark.parametrize("model", CLUSTERERS, ids=obj_name)
    def test_is_clusterer(self, model):
        """
        Test that is_clusterer works for instances and classes
        """
        assert inspect.isclass(model)
        assert is_clusterer(model)

        obj = model()
        assert is_clusterer(obj)

    @pytest.mark.parametrize(
        "model", REGRESSORS + CLASSIFIERS + TRANSFORMERS + DECOMPOSITIONS, ids=obj_name
    )
    def test_not_is_clusterer(self, model):
        """
        Test that is_clusterer does not match non-clusterer estimators
        """
        assert inspect.isclass(model)
        assert not is_clusterer(model)

        obj = model()
        assert not is_clusterer(obj)

    def test_clusterer_pipeline(self):
        """
        Test that is_clusterer works for pipelines
        """
        assert not is_clusterer(Pipeline)
        assert not is_clusterer(FeatureUnion)

        model = Pipeline([("reduce_dim", PCA()), ("kmeans", KMeans())])

        assert is_clusterer(model)

    @pytest.mark.parametrize(
        "viz,params", [
            (ModelVisualizer, {"estimator": KMeans()})
        ], ids=["ModelVisualizer"]
    )
    def test_is_clusterer_visualizer(self, viz, params):
        """
        Test that is_clusterer works on visualizers
        """
        assert inspect.isclass(viz)
        assert not is_clusterer(viz)

        obj = viz(**params)
        assert is_clusterer(obj)

    ##////////////////////////////////////////////////////////////////////
    ## is_gridsearch testing
    ##////////////////////////////////////////////////////////////////////

    def test_gridsearch_alias(self):
        """
        Assert isgridsearch aliases is_gridsearch
        """
        assert isgridsearch is is_gridsearch

    @pytest.mark.parametrize("model", SEARCH, ids=obj_name)
    def test_is_gridsearch(self, model):
        """
        Test that is_gridsearch works correctly
        """
        assert inspect.isclass(model)
        assert is_gridsearch(model)

        obj = model(SVC, {"C": [0.5, 1, 10]})
        assert is_gridsearch(obj)

    @pytest.mark.parametrize(
        "model", [MLPRegressor, MLPClassifier, SimpleImputer], ids=obj_name
    )
    def test_not_is_gridsearch(self, model):
        """
        Test that is_gridsearch does not match non grid searches
        """
        assert inspect.isclass(model)
        assert not is_gridsearch(model)

        obj = model()
        assert not is_gridsearch(obj)

    ##////////////////////////////////////////////////////////////////////
    ## is_probabilistic testing
    ##////////////////////////////////////////////////////////////////////

    def test_probabilistic_alias(self):
        """
        Assert isprobabilistic aliases is_probabilistic
        """
        assert isprobabilistic is is_probabilistic

    @pytest.mark.parametrize(
        "model",
        [
            MultinomialNB,
            GaussianNB,
            LogisticRegression,
            SVC,
            RandomForestClassifier,
            GradientBoostingClassifier,
            MLPClassifier,
        ],
        ids=obj_name,
    )
    def test_is_probabilistic(self, model):
        """
        Test that is_probabilistic works correctly
        """
        assert inspect.isclass(model)
        assert is_probabilistic(model)

        obj = model()
        assert is_probabilistic(obj)

    @pytest.mark.parametrize(
        "model",
        [MLPRegressor, SimpleImputer, StandardScaler, KMeans, RandomForestRegressor],
        ids=obj_name,
    )
    def test_not_is_probabilistic(self, model):
        """
        Test that is_probabilistic does not match non probablistic estimators
        """
        assert inspect.isclass(model)
        assert not is_probabilistic(model)

        obj = model()
        assert not is_probabilistic(obj)


##########################################################################
## Data type checking test cases
##########################################################################


class TestDataTypeChecking(object):
    """
    Test data type checking utilities
    """

    ##////////////////////////////////////////////////////////////////////
    ## is_dataframe testing
    ##////////////////////////////////////////////////////////////////////

    def test_dataframe_alias(self):
        """
        Assert isdataframe aliases is_dataframe
        """
        assert isdataframe is is_dataframe

    @pytest.mark.skipif(pd is None, reason="requires pandas")
    def test_is_dataframe(self):
        """
        Test that is_dataframe works correctly
        """
        df = pd.DataFrame(
            [{"a": 1, "b": 2.3, "c": "Hello"}, {"a": 2, "b": 3.14, "c": "World"}]
        )

        assert is_dataframe(df)

    @pytest.mark.parametrize(
        "obj",
        [
            np.array(
                [(1, 2.0, "Hello"), (2, 3.0, "World")],
                dtype=[("foo", "i4"), ("bar", "f4"), ("baz", "S10")],
            ),
            np.array([[1, 2, 3], [1, 2, 3]]),
            [[1, 2, 3], [1, 2, 3]],
        ],
        ids=["structured array", "array", "list"],
    )
    def test_not_is_dataframe(self, obj):
        """
        Test that is_dataframe does not match non-dataframes
        """
        assert not is_dataframe(obj)

    ##////////////////////////////////////////////////////////////////////
    ## is_series testing
    ##////////////////////////////////////////////////////////////////////

    def test_series_alias(self):
        """
        Assert isseries aliases is_series
        """
        assert isseries is is_series

    @pytest.mark.skipif(pd is None, reason="requires pandas")
    def test_is_series(self):
        """
        Test that is_series works correctly
        """
        df = pd.Series([1, 2, 3])

        assert is_series(df)

    @pytest.mark.parametrize(
        "obj",
        [
            np.array(
                [(1, 2.0, "Hello"), (2, 3.0, "World")],
                dtype=[("foo", "i4"), ("bar", "f4"), ("baz", "S10")],
            ),
            np.array([1, 2, 3]),
            [1, 2, 3],
        ],
        ids=["structured array", "array", "list"],
    )
    def test_not_is_series(self, obj):
        """
        Test that is_series does not match non-dataframes
        """
        assert not is_series(obj)

    ##////////////////////////////////////////////////////////////////////
    ## is_structured_array testing
    ##////////////////////////////////////////////////////////////////////

    def test_structured_array_alias(self):
        """
        Assert isstructuredarray aliases is_structured_array
        """
        assert isstructuredarray is is_structured_array

    def test_is_structured_array(self):
        """
        Test that is_structured_array works correctly
        """
        x = np.array(
            [(1, 2.0, "Hello"), (2, 3.0, "World")],
            dtype=[("foo", "i4"), ("bar", "f4"), ("baz", "S10")],
        )

        assert is_structured_array(x)

    @pytest.mark.parametrize(
        "obj", [np.array([[1, 2, 3], [1, 2, 3]]), [[1, 2, 3], [1, 2, 3]]], ids=obj_name
    )
    def test_not_is_structured_array(self, obj):
        """
        Test that is_structured_array does not match non-structured-arrays
        """
        assert not is_structured_array(obj)
