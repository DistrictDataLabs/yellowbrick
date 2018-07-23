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
    affiliation: 1
  - name: Rebecca Bilbro
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: University of Maryland
   index: 1
date: 23 July 2018
bibliography: paper.bib
---

# Summary

Proposed outline (let's go for the full 1k words):

1. Introduce the model selection triple
2. An argument for visual steering
3. Describe the scikit-learn API
4. Extending Estimator objects with a visualizer
5. Why Yellowbrick is better than the pyplot API
6. Integrating visual diagnostics into the ML workflow
7. Examples of Feature Analysis, Regression, Classification, and Clustering

# Acknowledgements

Since we first introduced the idea of Yellowbrick at PyCon 2016, several people have joined us in research labs and have stuck with us through 12 releases, ensuring the success of the project. Nathan Danielsen joined very early on and was one of our first maintainers, bringing an engineering perspective to our work and giving us much needed stability in testing. Larry Gray, Neal Humphrey, Jason Keung, Prema Roman, Kristen McIntyre, Jessica D'Amico and Adam Morris have also all joined our project as maintainers and core contributors, and we can't thank them enough.

Yellowbrick would not be possible without the invaluable contributions of those in the Python and Data Science community. At the time of this writing, GitHub reports that 44 contributors have submitted pull requests that have been merged and released and we expect this number to continue to grow. Every week, users submit feature requests, bug reports, suggestions and questions that allow us to make the software better and more robust. Other users write blog posts about their experience with Yellowbrick, encouraging both those new to machine learning and those that are old hat to more fully understand the models they are fitting. We can't thank the community enough for their support and their ongoing participation.

# References
