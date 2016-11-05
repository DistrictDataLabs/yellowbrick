# Yellowbrick

[![Build Status](https://travis-ci.org/DistrictDataLabs/yellowbrick.svg?branch=master)](https://travis-ci.org/DistrictDataLabs/yellowbrick)
[![Coverage Status](https://coveralls.io/repos/github/DistrictDataLabs/yellowbrick/badge.svg?branch=master)](https://coveralls.io/github/DistrictDataLabs/yellowbrick?branch=master)
[![Code Health](https://landscape.io/github/DistrictDataLabs/yellowbrick/master/landscape.svg?style=flat)](https://landscape.io/github/DistrictDataLabs/yellowbrick/master)
[![Documentation Status](https://readthedocs.org/projects/yellowbrick/badge/?version=latest)](http://yellowbrick.readthedocs.io/en/latest/?badge=latest)
[![Stories in Ready](https://badge.waffle.io/DistrictDataLabs/yellowbrick.png?label=ready&title=Ready)](https://waffle.io/DistrictDataLabs/yellowbrick)


A suite of visual analysis and diagnostic tools to facilitate feature selection, model selection, and parameter tuning for machine learning.


![Follow the yellow brick road](docs/images/yellowbrickroad.jpg)
Image by [Quatro Cinco](https://flic.kr/p/2Yj9mj), used with permission, Flickr Creative Commons.

# What is Yellowbrick?
Yellowbrick is a suite of visual analysis and diagnostic tools to facilitate feature selection, model selection, and parameter tuning for machine learning. All visualizations are generated in Matplotlib. Custom `yellowbrick` visualization tools include:

## Tools for feature analysis and selection
 - Boxplots (box-and-whisker plots)    
 - Violinplots    
 - Histograms    
 - Scatter plot matrices (sploms)    
 - Radial visualizations (radviz)    
 - Parallel coordinates    
 - Jointplots    
 - Rank 1D    
 - Rank 2D        

## Tools for model evaluation
### Classification
 - ROC-AUC curves    
 - Classification heatmaps
 - Class balance chart     

### Regression
 - Prediction error plots     
 - Residual plots     
 - Most informative features    

### Clustering
 - Silhouettes  
 - Density measures     

## Tools for parameter tuning
 - Validation curves    
 - Gridsearch heatmaps    

## Using Yellowbrick
The Yellowbrick API is specifically designed to play nicely with Scikit-Learn. Here is an example of a typical workflow sequence with Scikit-Learn and Yellowbrick:

### Feature Visualization
In this example, we see how Rank2D performs pairwise comparisons of each feature in the data set with a specific metric or algorithm, then returns them ranked as a lower left triangle diagram.
```python
from yellowbrick.features import Rank2D

visualizer = Rank2D(features=features, algorithm='covariance')
visualizer.fit(X, y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()                   # Draw/show/poof the data
```

### Model Visualization
In this example, we instantiate a Scikit-Learn classifier, and then we use Yellowbrick's ROCAUC class to visualize the tradeoff between the classifier's sensitivity and specificity.
```python
from sklearn.svm import LinearSVC
from yellowbrick import ROCAUC

model = LinearSVC()
model.fit(X,y)
y_pred = model.predict(X)
visualizer = ROCAUC(model)
visualizer.score(y,y_pred)
visualizer.poof()
```

For additional information on getting started with Yellowbrick, check out our [examples notebook](https://github.com/DistrictDataLabs/yellowbrick/blob/develop/examples/examples.ipynb).

We also have a [quick start guide](https://github.com/DistrictDataLabs/yellowbrick/blob/master/docs/setup.rst).

## Contributing to Yellowbrick

Yellowbrick is an open source tool designed to enable more informed machine learning through visualizations. If you would like to contribute, you can do so in the following ways:

 - Add issues or bugs to the bug tracker: https://github.com/DistrictDataLabs/yellowbrick/issues
 - Work on a card on the dev board: https://waffle.io/DistrictDataLabs/yellowbrick
 - Create a pull request in Github: https://github.com/DistrictDataLabs/yellowbrick/pulls

This repository is set up in a typical production/release/development cycle as described in [A Successful Git Branching Model](http://nvie.com/posts/a-successful-git-branching-model/). A typical workflow is as follows:

1. Select a card from the [dev board](https://waffle.io/districtdatalabs/yellowbrick) - preferably one that is "ready" then move it to "in-progress".    
2. Create a branch off of develop called "feature-[feature name]", work and commit into that branch.
    ```
    ~$ git checkout -b feature-myfeature develop
    ```   

3. Once you are done working (and everything is tested) merge your feature into develop.
    ```
    ~$ git checkout develop
    ~$ git merge --no-ff feature-myfeature
    ~$ git branch -d feature-myfeature
    ~$ git push origin develop
    ```

4. Repeat. Releases will be routinely pushed into master via release branches, then deployed to the server.
