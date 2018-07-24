---
title: 'Yellowbrick: Visualizing the Scikit-Learn Model Selection Process'
tags:
  - machine learning
  - visual analysis
  - model selection
  - Python
  - scikit-learn
  - matplotlib
authors:
  - name: Benjamin Bengfort
    orcid: 0000-0003-0660-7682
    affiliation: "1, 2"
  - name: Rebecca Bilbro
    orcid: 0000-0002-1143-044X
    affiliation: 2
affiliations:
 - name: University of Maryland
   index: 1
 - name: Georgetown University
   index: 2
date: 23 July 2018
bibliography: paper.bib
---

# Summary

Discussions of machine learning are frequently characterized by a singular focus on model selection. Be it logistic regression, random forests, Bayesian methods, or artificial neural networks, machine learning practitioners are often quick to express their preference. However, model selection is a more nuanced than simply picking the “right” or “wrong” algorithm. In practice, the workflow includes multiple iterations through feature selection, algorithm selection, and hyperparameter tuning, first described as the model selection triple in a 2015 SIGMOD paper by Kumar et al. [@kumar2016model]. “Model selection,” they explain, “is iterative and exploratory because the space of [model selection triples] is usually infinite, and it is generally impossible for analysts to know a priori which [combination] will yield satisfactory accuracy and/or insights.”

Recently, much of this workflow has been automated through grid search methods, standardized APIs, and GUI-based applications. In practice, however, research in both machine learning [@wickham_visualizing_2015] and visual analytics [@liu_wang_liu_zhu_2017] suggests human intuition and guidance can more effectively hone in on quality models than exhaustive search. By visualizing the model selection process, data scientists can steer towards final, explainable models and avoid pitfalls and traps.

The Yellowbrick library is a diagnostic visualization platform for machine learning that allows data scientists to steer the model selection process. For users, Yellowbrick can help evaluate the performance, stability, and predictive value of machine learning models and assist in diagnosing problems throughout the machine learning workflow. Yellowbrick achieves this visual steering by extending both scikit-learn [@scikit-learn] and Matplotlib [@matplotlib].

scikit-learn is an extension of SciPy whose primary purpose is to provide machine learning algorithms as well as the tools and utilities required to engage in successful modeling. Its primary contribution is an object-oriented "API for machine learning" that exposes the implementations of a wide array of model families into a hierarchy of interfaces for different machine learning tasks. The root of the hierarchy is an Estimator, broadly any object that can learn from data. The primary Estimator objects implement classifiers, regressors, or clustering algorithms. Scikit-Learn also specifies utilities for performing machine learning, in particular Transformers, is a special type of Estimator that creates a new dataset from an old one based on rules that it has learned from the fitting process.

All scikit-learn estimators have a fit(X, y=None) method that accepts a two dimensional data array, X, and optionally a vector y of target values. The fit() method trains the estimator, making it ready to transform data or make predictions. Transformers have an associated transform(X) method that returns a new dataset, Xprime and models have a predict(X) method that returns a vector of predictions, yhat. Models also have a score(X, y) method that evaluate the performance of the model.

Yellowbrick extends the scikit-learn API with a new core object, the Visualizer, which provies visual diagnostics throughout the transformation of high-dimensional data, allowing visual models to be fit and transformed as part of the scikit-learn pipeline process. Visualizers interact with scikit-learn objects to perform actions related to fit(), transform(), predict(), and score().  

The Yellowbrick API also wraps Matplotlib to create publication-ready figures and interactive data explorations while still allowing developers fine-grain control of figures. After a visualizer has been fit and scored or fit and transformed, the draw() method initializes the underlying figure associated with the visualizer. The user calls the visualizer’s poof() method, which in turn calls a finalize() method on the visualizer to draw legends, titles, etc. and then poof() renders the figure.

<!--- Add Examples of Feature Analysis, Regression, Classification, and Clustering -->

# Acknowledgements

Since we first introduced the idea of Yellowbrick at PyCon 2016, several people have joined us in research labs and have stuck with us through 12 releases, ensuring the success of the project. Nathan Danielsen joined very early on and was one of our first maintainers, bringing an engineering perspective to our work and giving us much needed stability in testing. Larry Gray, Neal Humphrey, Jason Keung, Prema Roman, Kristen McIntyre, Jessica D'Amico and Adam Morris have also all joined our project as maintainers and core contributors, and we can't thank them enough.

Yellowbrick would not be possible without the invaluable contributions of those in the Python and Data Science community. At the time of this writing, GitHub reports that 44 contributors have submitted pull requests that have been merged and released and we expect this number to continue to grow. Every week, users submit feature requests, bug reports, suggestions and questions that allow us to make the software better and more robust. Other users write blog posts about their experience with Yellowbrick, encouraging both those new to machine learning and those that are old hat to more fully understand the models they are fitting. We can't thank the community enough for their support and their ongoing participation.

# References
