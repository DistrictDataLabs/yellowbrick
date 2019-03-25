.. -*- mode: rst -*-

Residuals Plot
==============

Residuals, in the context of regression models, are the difference between the observed value of the target variable (y) and the predicted value (ŷ), i.e. the error of the prediction. The residuals plot shows the difference between residuals on the vertical axis and the dependent variable on the horizontal axis, allowing you to detect regions within the target that may be susceptible to more or less error.

.. plot::
    :context: close-figs
    :alt: Residuals Plot on the Concrete dataset using a linear model

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import ResidualsPlot

    # Load a regression dataset
    X, y = load_concrete()

    # Create the train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the linear model and visualizer
    model = Ridge()
    visualizer = ResidualsPlot(model)

    visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.poof()                 # Draw/show/poof the data


A common use of the residuals plot is to analyze the variance of the error of the regressor. If the points are randomly dispersed around the horizontal axis, a linear regression model is usually appropriate for the data; otherwise, a non-linear model is more appropriate. In the case above, we see a fairly random, uniform distribution of the residuals against the target in two dimensions. This seems to indicate that our linear model is performing well. We can also see from the histogram that our error is normally distributed around zero, which also generally indicates a well fitted model.

Note that if the histogram is not desired, it can be turned off with the ``hist=False`` flag:

.. plot::
    :context: close-figs
    :alt: Residuals Plot on the Concrete dataset without a histogram

    visualizer = ResidualsPlot(model, hist=False)
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.poof()

.. warning:: The histogram on the residuals plot requires matplotlib 2.0.2 or greater. If you are using an earlier version of matplotlib, simply set the ``hist=False`` flag so that the histogram is not drawn.

API Reference
-------------

.. automodule:: yellowbrick.regressor.residuals
    :members: ResidualsPlot
    :undoc-members:
    :show-inheritance:
