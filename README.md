# Yellowbrick

[![Build Status](https://travis-ci.org/DistrictDataLabs/yellowbrick.svg?branch=master)](https://travis-ci.org/DistrictDataLabs/yellowbrick)
[![Coverage Status](https://coveralls.io/repos/github/DistrictDataLabs/yellowbrick/badge.svg?branch=master)](https://coveralls.io/github/DistrictDataLabs/yellowbrick?branch=master)
[![Code Health](https://landscape.io/github/DistrictDataLabs/yellowbrick/master/landscape.svg?style=flat)](https://landscape.io/github/DistrictDataLabs/yellowbrick/master)
[![Documentation Status](https://readthedocs.org/projects/yellowbrick/badge/?version=latest)](http://yellowbrick.readthedocs.io/en/latest/?badge=latest)
[![Stories in Ready](https://badge.waffle.io/DistrictDataLabs/yellowbrick.png?label=ready&title=Ready)](https://waffle.io/DistrictDataLabs/yellowbrick)


A suite of visual analysis and diagnostic tools to facilitate feature selection, model selection, and parameter tuning for machine learning.


![Follow the yellow brick road](images/yellowbrickroad.jpg)
Image by [Quatro Cinco](https://flic.kr/p/2Yj9mj), used with permission, Flickr Creative Commons.

# What is Yellowbrick?
Yellowbrick is a suite of visual analysis and diagnostic tools to facilitate feature selection, model selection, and parameter tuning for machine learning. All visualizations are generated in Matplotlib. Custom `yellowbrick` visualization tools include:

## Tools for feature analysis and selection
- boxplots (box-and-whisker plots)    
- violinplots    
- histograms    
- scatter plot matrices (sploms)    
- radial visualizations (radviz)    
- parallel coordinates    
- jointplots    
- diagonal correlation matrix    

## Tools for model evaluation
### Classification
- ROC curves    
- classification heatmaps    

### Regression
- prediction error plots     
- residual plots     

## Tools for parameter tuning
- validation curves    
- gridsearch heatmap    

## Using Yellowbrick
For information on getting started with Yellowbrick, check out our [quick start guide](https://github.com/DistrictDataLabs/yellowbrick/blob/develop/docs/setup.md).

## Contributing to Yellowbrick

Yellowbrick is an open source tool designed to enable more informed machine learning through visualizations. If you would like to contribute, you can do so in the following ways:

 - Add issues or bugs to the bug tracker: https://github.com/DistrictDataLabs/yellowbrick/issues
 - Work on a card on the dev board: https://waffle.io/DistrictDataLabs/yellowbrick
 - Create a pull request in Github: https://github.com/DistrictDataLabs/yellowbrick/pulls

This repository is set up in a typical production/release/development cycle as described in [A Successful Git Branching Model](http://nvie.com/posts/a-successful-git-branching-model/). A typical workflow is as follows:

1. Select a card from the [dev board](https://waffle.io/districtdatalabs/yellowbrick) - preferably one that is "ready" then move it to "in-progress".    
2. Create a branch off of develop called "feature-[feature name]", work and commit into that branch.
```bash
~$ git checkout -b feature-myfeature develop
```    
3. Once you are done working (and everything is tested) merge your feature into develop.
```bash
~$ git checkout develop
~$ git merge --no-ff feature-myfeature
~$ git branch -d feature-myfeature
~$ git push origin develop
```    
4. Repeat. Releases will be routinely pushed into master via release branches, then deployed to the server.
