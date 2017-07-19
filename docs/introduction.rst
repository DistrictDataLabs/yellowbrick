============
Introduction
============

.. image:: images/yellowbrickroad.jpg

Image by QuatroCinco_, used with permission, Flickr Creative Commons.

Yellowbrick is a new Python library that extends the Scikit-Learn API_ to incorporate visualizations into the machine learning workflow.

Yellowbrick is an open source, pure Python project that extends Scikit-Learn with visual analysis and diagnostic tools. The Yellowbrick API also wraps Matplotlib to create publication-ready figures and interactive data explorations while still allowing developers fine-grain control of figures. For users, Yellowbrick can help evaluate the performance, stability, and predictive value of machine learning models, and assist in diagnosing problems throughout the machine learning workflow.

Recently, much of this workflow has been automated through grid search methods, standardized APIs, and GUI-based applications. In practice, however, human intuition and guidance can more effectively hone in on quality models than exhaustive search. By visualizing the model selection process, data scientists can steer towards final, explainable models and avoid pitfalls and traps.

The Yellowbrick library is a diagnostic visualization platform for machine learning that allows data scientists to steer the model selection process. Yellowbrick extends the Scikit-Learn API with a new core object: the Visualizer. Visualizers allow visual models to be fit and transformed as part of the Scikit-Learn Pipeline process, providing visual diagnostics throughout the transformation of high dimensional data.

Presentations
-------------

Yellowbrick has enjoyed the spotlight at a few conferences and in several presentations. We hope that these videos, talks, and slides will help you understand Yellowbrick a bit better.

Videos:
    - `Visual Diagnostics for More Informed Machine Learning: Within and Beyond Scikit-Learn (PyCon 2016) <https://youtu.be/c5DaaGZWQqY>`_
    - `Visual Diagnostics for More Informed Machine Learning (PyData Carolinas 2016) <https://youtu.be/cgtNPx7fJUM>`_

Slides:
    - `Visualizing the Model Selection Process <https://www.slideshare.net/BenjaminBengfort/visualizing-the-model-selection-process>`_
    - `Visualizing Model Selection with Scikit-Yellowbrick <https://www.slideshare.net/BenjaminBengfort/visualizing-model-selection-with-scikityellowbrick-an-introduction-to-developing-visualizers>`_

.. _QuatroCinco: https://flic.kr/p/2Yj9mj
.. _API: http://scikit-learn.org/stable/modules/classes.html
