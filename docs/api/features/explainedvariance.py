#!/usr/bin/env python3

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from yellowbrick.datasets import load_concrete
    from yellowbrick.features import ExplainedVariance

    # Load the dataset
    X, y = load_concrete()

    # Plotting individual variances + Kaiser
    viz = ExplainedVariance(kaiser=True)
    viz.fit(X)              # Fit the data to the visualizer
    viz.transform(X)        # Transform the data
    viz.poof("images/explainedvariance_kaiser.png")
    
    plt.clf()
    
    # Scree plot
    viz2 = ExplainedVariance(scree=True)
    viz2.fit(X)
    viz2.transform(X)
    viz2.poof("images/explainedvariance_scree.png")

