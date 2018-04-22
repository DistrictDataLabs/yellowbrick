About
=====

.. image:: images/yellowbrickroad.jpg

Image by QuatroCinco_, used with permission, Flickr Creative Commons.

Yellowbrick is an open source, pure Python project that extends the scikit-learn API_ with visual analysis and diagnostic tools. The Yellowbrick API also wraps matplotlib to create publication-ready figures and interactive data explorations while still allowing developers fine-grain control of figures. For users, Yellowbrick can help evaluate the performance, stability, and predictive value of machine learning models and assist in diagnosing problems throughout the machine learning workflow.

Recently, much of this workflow has been automated through grid search methods, standardized APIs, and GUI-based applications. In practice, however, human intuition and guidance can more effectively hone in on quality models than exhaustive search. By visualizing the model selection process, data scientists can steer towards final, explainable models and avoid pitfalls and traps.

The Yellowbrick library is a diagnostic visualization platform for machine learning that allows data scientists to steer the model selection process. It extends the scikit-learn API with a new core object: the Visualizer. Visualizers allow visual models to be fit and transformed as part of the scikit-learn pipeline process, providing visual diagnostics throughout the transformation of high-dimensional data.

Model Selection
---------------
Discussions of machine learning are frequently characterized by a singular focus on model selection. Be it logistic regression, random forests, Bayesian methods, or artificial neural networks, machine learning practitioners are often quick to express their preference. The reason for this is mostly historical. Though modern third-party machine learning libraries have made the deployment of multiple models appear nearly trivial, traditionally the application and tuning of even one of these algorithms required many years of study. As a result, machine learning practitioners tended to have strong preferences for particular (and likely more familiar) models over others.

However, model selection is a bit more nuanced than simply picking the "right" or "wrong" algorithm. In practice, the workflow includes:

  1. selecting and/or engineering the smallest and most predictive feature set
  2. choosing a set of algorithms from a model family
  3. tuning the algorithm hyperparameters to optimize performance

The **model selection triple** was first described in a 2015 SIGMOD_ paper by Kumar et al. In their paper, which concerns the development of next-generation database systems built to anticipate predictive modeling, the authors cogently express that such systems are badly needed due to the highly experimental nature of machine learning in practice. "Model selection," they explain, "is iterative and exploratory because the space of [model selection triples] is usually infinite, and it is generally impossible for analysts to know a priori which [combination] will yield satisfactory accuracy and/or insights."

Name Origin
-----------
The Yellowbrick package gets its name from the fictional element in the 1900 children's novel **The Wonderful Wizard of Oz** by American author L. Frank Baum. In the book, the yellow brick road is the path that the protagonist, Dorothy Gale, must travel in order to reach her destination in the Emerald City.

From Wikipedia_:
    "The road is first introduced in the third chapter of The Wonderful Wizard of Oz. The road begins in the heart of the eastern quadrant called Munchkin Country in the Land of Oz. It functions as a guideline that leads all who follow it, to the road's ultimate destination—the imperial capital of Oz called Emerald City that is located in the exact center of the entire continent. In the book, the novel's main protagonist, Dorothy, is forced to search for the road before she can begin her quest to seek the Wizard. This is because the cyclone from Kansas did not release her farmhouse closely near it as it did in the various film adaptations. After the council with the native Munchkins and their dear friend the Good Witch of the North, Dorothy begins looking for it and sees many pathways and roads nearby, (all of which lead in various directions). Thankfully it doesn't take her too long to spot the one paved with bright yellow bricks."

Team
----

Yellowbrick is developed by data scientists who believe in open source and the project enjoys contributions from Python developers all over the world. The project was started by `@rebeccabilbro`_ and `@bbengfort`_ as an attempt to better explain machine learning concepts to their students; they quickly realized, however, that the potential for visual steering could have a large impact on practical data science and developed it into a high-level Python library.

Yellowbrick is incubated by `District Data Labs`_, an organization that is dedicated to collaboration and open source development. As part of District Data Labs, Yellowbrick was first introduced to the Python Community at `PyCon 2016 <https://youtu.be/c5DaaGZWQqY>`_ in both talks and during the development sprints. The project was then carried on through DDL Research Labs (semester-long sprints where members of the DDL community contribute to various data-related projects).

For a full list of current maintainers and core contributors, please see `MAINTAINERS.md <https://github.com/DistrictDataLabs/yellowbrick/blob/develop/MAINTAINERS.md>`_ in the root of our GitHub repository. Thank you so much to everyone who has `contributed to Yellowbrick <https://github.com/DistrictDataLabs/yellowbrick/graphs/contributors>`_!

License
-------

Yellowbrick is an open source project and its `license <https://github.com/DistrictDataLabs/yellowbrick/blob/master/LICENSE.txt>`_ is an implementation of the FOSS `Apache 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_ license by the Apache Software Foundation. `In plain English <https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)>`_ this means that you can use Yellowbrick for commercial purposes, modify and distribute the source code, and even sublicense it. We want you to use Yellowbrick, profit from it, and contribute back if you do cool things with it.

There are, however, a couple of requirements that we ask from you. First, when you copy or distribute Yellowbrick source code, please include our copyright and license found in the `LICENSE.txt <https://github.com/DistrictDataLabs/yellowbrick/blob/master/LICENSE.txt>`_ at the root of our software repository. In addition, if we create a file called "NOTICE" in our project you must also include that in your source distribution. The "NOTICE" file will include attribution and thanks to those who have worked so hard on the project! Finally you can't hold District Data Labs or any Yellowbrick contributor liable for your use of our software, nor use any of our names, trademarks, or logos.

We think that's a pretty fair deal, and we're big believers in open source. If you make any changes to our software, use it commercially or academically, or have any other interest, we'd love to hear about it.


.. _SIGMOD: http://cseweb.ucsd.edu/~arunkk/vision/SIGMODRecord15.pdf
.. _Wikipedia: https://en.wikipedia.org/wiki/Yellow_brick_road
.. _`@rebeccabilbro`: https://github.com/rebeccabilbro
.. _`@bbengfort`: https://github.com/bbengfort
.. _`District Data Labs`: http://www.districtdatalabs.com/

Presentations
-------------

Yellowbrick has enjoyed the spotlight in several presentations at recent conferences. We hope that these notebooks, talks, and slides will help you understand Yellowbrick a bit better.

Conference Presentations (videos):
    - `Visual Diagnostics for More Informed Machine Learning: Within and Beyond Scikit-Learn (PyCon 2016) <https://youtu.be/c5DaaGZWQqY>`_
    - `Yellowbrick: Steering Machine Learning with Visual Transformers (PyData London 2017) <https://youtu.be/2ZKng7pCB5k>`_

Jupyter Notebooks:
    - `Data Science Delivered: ML Regression Predications <https://github.com/ianozsvald/data_science_delivered/blob/master/ml_explain_regression_prediction.ipynb>`_

Slides:
    - `Machine Learning Libraries You'd Wish You'd Known About (PyData Budapest 2017) <https://speakerdeck.com/ianozsvald/machine-learning-libraries-youd-wish-youd-known-about-1>`_ 
    - `Visualizing the Model Selection Process <https://www.slideshare.net/BenjaminBengfort/visualizing-the-model-selection-process>`_
    - `Visualizing Model Selection with Scikit-Yellowbrick <https://www.slideshare.net/BenjaminBengfort/visualizing-model-selection-with-scikityellowbrick-an-introduction-to-developing-visualizers>`_
    - `Visual Pipelines for Text Analysis (Data Intelligence 2017) <https://speakerdeck.com/dataintelligence/visual-pipelines-for-text-analysis>`_

.. _QuatroCinco: https://flic.kr/p/2Yj9mj
.. _API: http://scikit-learn.org/stable/modules/classes.html


Citing Yellowbrick
------------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1206239.svg
   :target: https://doi.org/10.5281/zenodo.1206239

We hope that Yellowbrick facilitates machine learning of all kinds and we're particularly fond of academic work and research. If you're writing a scientific publication that uses Yellowbrick you can cite *Bengfort et al. (2018)* with the following BibTex:

.. code-block:: bibtex

    @software{bengfort_yellowbrick_2018,
        title = {Yellowbrick},
        rights = {Apache License 2.0},
        url = {http://www.scikit-yb.org/en/latest/},
        abstract = {Yellowbrick is an open source, pure Python project that
            extends the Scikit-Learn {API} with visual analysis and
            diagnostic tools. The Yellowbrick {API} also wraps Matplotlib to
            create publication-ready figures and interactive data
            explorations while still allowing developers fine-grain control
            of figures. For users, Yellowbrick can help evaluate the
            performance, stability, and predictive value of machine learning
            models, and assist in diagnosing problems throughout the machine
            learning workflow.},
        version = {0.6},
        author = {Bengfort, Benjamin and Danielsen, Nathan and
            Bilbro, Rebecca and Gray, Larry and {McIntyre}, Kristen and
            Richardson, George and Miller, Taylor and Mayfield, Gary and
            Schafer, Phillip and Keung, Jason},
        date = {2018-03-17},
        doi = {10.5281/zenodo.1206264}
    }

You can also find DOI (digital object identifiers) for every version of Yellowbrick on `zenodo.org <https://doi.org/10.5281/zenodo.1206239>`_; use the BibTeX on this site to reference specific versions or changes made to the software.

We're also currently working on a scientific paper that describes Yellowbrick in the context of *steering the model selection process*. Stay tuned for a pre-release of this paper on arXiv.

Contacting Us
-------------

The best way to contact the Yellowbrick team is to send us a note on one of the following platforms:

- Send an email via our `mailing list`_.
- Direct message us on `Twitter`_.
- Ask a question on `Stack Overflow`_.
- Report an issue on our `GitHub Repo`_.

.. _`GitHub Repo`: https://github.com/DistrictDataLabs/yellowbrick
.. _`mailing list`: http://bit.ly/yb-listserv
.. _`Stack Overflow`: https://stackoverflow.com/questions/tagged/yellowbrick
.. _`Twitter`: https://twitter.com/scikit_yb
