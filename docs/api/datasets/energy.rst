.. -*- mode: rst -*-

Energy
======

The dataset was created by Angeliki Xifara (angxifara '@' gmail.com, Civil/Structural Engineer) and was processed by Athanasios Tsanas (tsanasthanasis '@' gmail.com, Oxford Centre for Industrial and Applied Mathematics, University of Oxford, UK).

=================    ==========================
Samples total                               768
Dimensionality                                8
Features                              real, int
Targets                        float, 6.01-43.1
Task(s)              regression, classification
=================    ==========================

Description
-----------

We perform energy analysis using 12 different building shapes simulated in Ecotect. The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters. We simulate various settings as functions of the afore-mentioned characteristics to obtain 768 building shapes. The dataset comprises 768 samples and 8 features, aiming to predict two real valued responses. It can also be used as a multi-class classification problem if the response is rounded to the nearest integer.

Example
-------

The energy dataset contains a multi-target supervised dataset for both the heating and the cooling load of buildings. By default only the heating load is returned for most examples. To perform a multi-target regression, simply access the dataframe and select both the heating and cooling load columns as follows:

.. code:: python

    from yellowbrick.datasets import load_energy
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split as tts

    features = [
       "relative compactness",
       "surface area",
       "wall area",
       "roof area",
       "overall height",
       "orientation",
       "glazing area",
       "glazing area distribution",
    ]
    target = ["heating load", "cooling load"]

    df = load_energy(return_dataset=True).to_dataframe()
    X, y = df[features], df[target]

    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)

    model = RandomForestRegressor().fit(X_train, y_train)
    model.score(X_test, y_test)


Note that not all regressors support multi-target regression, one simple strategy in this case is to use a :class:`sklearn.multioutput.MultiOutputRegressor`, which fits an estimator for each target.


Citation
--------

Downloaded from the `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml/datasets/Energy+efficiency>`_ March 23, 2015.

A. Tsanas, A. Xifara: 'Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools', Energy and Buildings, Vol. 49, pp. 560-567, 2012

For further details on the data analysis methodology:

A. Tsanas, 'Accurate telemonitoring of Parkinson's disease symptom severity using nonlinear speech signal processing and statistical machine learning', D.Phil. thesis, University of Oxford, 2012

Loader
------

.. autofunction:: yellowbrick.datasets.loaders.load_energy
