---
title: 'Yellowbrick: Visualizing the Scikit-Learn Model Selection Process'
tags:
  - machine learning
  - visual analysis
  - model selection
  - python
  - scikit-learn
  - matplotlib
authors:
  - name: Benjamin Bengfort
    orcid: 0000-0003-0660-7682
    affiliation: 1
  - name: Rebecca Bilbro
    orcid: 0000-0002-1143-044X
    affiliation: 1
affiliations:
 - name: Georgetown University
   index: 1
date: 30 July 2018
bibliography: paper.bib
---

# Summary

Discussions of machine learning are frequently characterized by a singular focus on algorithmic behavior. Be it logistic regression, random forests, Bayesian methods, or artificial neural networks, practitioners are often quick to express their preference. However, model selection is more nuanced than simply picking the “right” or “wrong” algorithm. In practice, the workflow includes multiple iterations through feature engineering, algorithm selection, and hyperparameter tuning &mdash; summarized by Kumar et al. as a search for the maximally performing model selection triple [@kumar2016model]. “Model selection,” they explain, “is iterative and exploratory because the space of [model selection triples] is usually infinite, and it is generally impossible for analysts to know a priori which [combination] will yield satisfactory accuracy and/or insights.”

Treating model selection as search has led to automation through grid search methods, standardized APIs, drag and drop GUIs, and specialized database systems. However, the search problem is computationally intractable and research in both machine learning [@wickham_visualizing_2015] and visual analytics [@liu_wang_liu_zhu_2017] suggests human intuition and guidance can more effectively hone in on quality models than exhaustive optimization methods. By visualizing the model selection process, data scientists can interactively steer towards final, interpretable models and avoid pitfalls and traps [@kapoor2010interactive].

Yellowbrick is a response to the call for open source visual steering tools. For data scientists, Yellowbrick helps evaluate the stability and predictive value of machine learning models and improves the speed of the experimental workflow. For data engineers, Yellowbrick provides visual tools for monitoring model performance in real world applications. For users of models, Yellowbrick provides visual interpretation of the behavior of the model in high dimensional feature space. Finally, for students, Yellowbrick is a framework for understanding a large variety of algorithms and methods.

Implemented in Python, the Yellowbrick visualization package achieves steering by extending both scikit-learn [@sklearn] and Matplotlib [@matplotlib]. Like Yellowbrick, both scikit-learn and Matplotlib are extensions of SciPy [@scipy], libraries intended to facilitate scientific computing. Scikit-learn provides a generalized API for machine learning by exposing the concept of an `Estimator`, an object that learns from data. Yellowbrick in turn extends this concept with the idea of a `Visualizer`, an object that both learns from data and visualizes the result. Visualizers wrap Matplotlib procedures to produce publication-ready figures and rich visual analytics.

Because Yellowbrick is part of a rich visual and machine learning ecosystem, it provides visualizations for feature and target analysis, classification, regression, and clustering model visualization, hyperparameter tuning, and text analysis. A few selected examples of visual diagnostics for model selection and their interpretations follow.

![Feature Analysis](figures/feature_analysis.png)

Because “more data beats better algorithms” [@rajaraman2008more], the first step to creating valid, predictive models is to find the minimum set of features that predicts the dependent variable. Generally, this means finding features that describe data in high dimensional space that are *separable* (i.e., by a hyperplane). Tools like `RadViz`, `ParallelCoordinates`, and `Manifold` help visualize high dimensional data for quick diagnostics. Bayesian models and regressions suffer when independent variables are collinear (i.e., exhibit pairwise correlation). `Rank2D` visualizations show pairwise correlations among features and can facilitate feature elimination.

![Regression Model Tuning](figures/regression.png)

Regression models hypothesize some underlying function influenced by noise whose central tendency can be inferred. The `PredictionError` visualizer shows the relationship of actual to predicted values, giving a sense of heteroskedasticity in the target, or regions of more or less error as predictions deviate from the 45 degree line. The `ResidualsPlot` shows the relationship of error in the training and test data and can also show regions of increased variability in the predictive model.

![Classification Model Tuning](figures/classification.png)

Classification analysis focuses on the precision and recall of the model's prediction of individual classes. The `ClassificationReport` visualizer allows for rapid comparison between models as a visual heatmap of these metrics. The `DiscriminationThreshold` visualizer for binary classifiers shows how adjusting the threshold for positive classification may influence precision and recall globally, as well as the number of points that may require manual checking for stricter determination.

![Clustering Model Tuning](figures/clustering.png)

Searching for structure in unlabelled data can be challenging because evaluation is largely qualitative. When using K-Means models, choosing K has a large impact on the quality of the analysis; the `KElbowVisualizer` can help select the best K given computational constraints. The `SilhouetteVisualizer` shows the relationship of points in each cluster relative to other clusters and gives an overview of the composition and size of each cluster which may hint at how models group similar data points.

![Hyperparameter Tuning](figures/hyperparameter_tuning.png)

Yellowbrick also offers several other techniques for hyperparameter tuning. Model and regression-specific `AlphaSelection` visualizers help identify the impact of regularization on linear models and the influence of complexity on the trade-off between error due to bias or variance. More generally, the `LearningCurve` visualizer shows how sensitive models are to the amount of data the model is trained on.

Yellowbrick includes many more visualizations, intended to fit directly into the machine learning workflow, and many more are being added in each new release. From text analysis-specific visualizations to missing data analysis, to a `contrib` module that focuses on other machine learning libraries, Yellowbrick has tools to facilitate all parts of hypothesis driven development. The source code for Yellowbrick has been archived to Zenodo and the most recent version can be obtained with the linked DOI: [@zenodo].

# Acknowledgements

Since we first introduced the idea of Yellowbrick at PyCon 2016, many people have joined us and stuck with us through 12 releases, ensuring the success of the project. Nathan Danielsen joined very early on and was one of our first maintainers, bringing an engineering perspective to our work and giving us much needed stability in testing. Larry Gray, Neal Humphrey, Jason Keung, Prema Roman, Kristen McIntyre, Jessica D'Amico and Adam Morris have also all joined our project as maintainers and core contributors, and we can't thank them enough.

Yellowbrick would not be possible without the invaluable contributions of those in the Python and Data Science communities. At the time of this writing, GitHub reports that 46 contributors have submitted pull requests that have been merged and released, and we expect this number to continue to grow. Every week, users submit feature requests, bug reports, suggestions and questions that allow us to make the software better and more robust. Others write blog posts about using Yellowbrick, encouraging both newcomers and seasoned practitioners to more fully understand the models they are fitting. Our sincere thanks to the community for their ongoing support and participation.

# References
