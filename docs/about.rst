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


Who is Yellowbrick for?
-----------------------

Yellowbrick ``Visualizers`` have multiple use cases:

 - For data scientists, they can help evaluate the stability and predictive value of machine learning models and improve the speed of the experimental workflow.
 - For data engineers, Yellowbrick provides visual tools for monitoring model performance in real world applications.
 - For users of models, Yellowbrick provides visual interpretation of the behavior of the model in high dimensional feature space.
 - For teachers and students, Yellowbrick is a framework for teaching and understanding a large variety of algorithms and methods.


Name Origin
-----------
The Yellowbrick package gets its name from the fictional element in the 1900 children's novel **The Wonderful Wizard of Oz** by American author L. Frank Baum. In the book, the yellow brick road is the path that the protagonist, Dorothy Gale, must travel in order to reach her destination in the Emerald City.

From Wikipedia_:
    "The road is first introduced in the third chapter of The Wonderful Wizard of Oz. The road begins in the heart of the eastern quadrant called Munchkin Country in the Land of Oz. It functions as a guideline that leads all who follow it, to the road's ultimate destination—the imperial capital of Oz called Emerald City that is located in the exact center of the entire continent. In the book, the novel's main protagonist, Dorothy, is forced to search for the road before she can begin her quest to seek the Wizard. This is because the cyclone from Kansas did not release her farmhouse closely near it as it did in the various film adaptations. After the council with the native Munchkins and their dear friend the Good Witch of the North, Dorothy begins looking for it and sees many pathways and roads nearby, (all of which lead in various directions). Thankfully it doesn't take her too long to spot the one paved with bright yellow bricks."

Team
----

Yellowbrick is developed by volunteer data scientists who believe in open source and the project enjoys contributions from Python developers all over the world. The project was started by `@rebeccabilbro`_ and `@bbengfort`_ as an attempt to better explain machine learning concepts to their students at Georgetown University where they teach a data science certificate program. They quickly realized, however, that the potential for visual steering could have a large impact on practical data science and developed it into a production-ready Python library.

Yellowbrick was then incubated by District Data Labs (DDL) in partnership with Georgetown University. District Data Labs is an organization that is dedicated to open source development and data science education and provided resources to help Yellowbrick grow. Yellowbrick was first introduced to the Python Community at `PyCon 2016 <https://youtu.be/c5DaaGZWQqY>`_ in both talks and during the development sprints. The project was then carried on through DDL Research Labs -- semester-long sprints where members of the DDL community contribute to various data-related projects.

Since then, Yellowbrick has enjoyed the participation of a large number of contributors from around the world and growing support in the PyData community. Yellowbrick has been featured in talks at PyData, Scipy, NumFOCUS, and PSF organized events as well as blog posts and Kaggle competitions. We are so thrilled to have such a dedicated community involved in active contributions both large and small.

For a full list of current maintainers and core contributors, please see `MAINTAINERS.md <https://github.com/DistrictDataLabs/yellowbrick/blob/develop/MAINTAINERS.md>`_ in the root of our GitHub repository. Thank you so much to everyone who has `contributed to Yellowbrick <https://github.com/DistrictDataLabs/yellowbrick/graphs/contributors>`_!

Affiliations
------------

Yellowbrick is proud to be affiliated with several organizations that provide institutional support to the project. Such support is sometimes financial, often material, and always in the spirit of free and open source software. We can't thank them enough for their role in making Yellowbrick what it is today.

`District Data Labs`_: District Data Labs incubated Yellowbrick and sponsors research labs by purchasing food and organizing events. Research labs are semester long sprints that allow Yellowbrick contributors to meet in person, share a meal, and hack on the project. DDL also sponsors travel to PyCon and PyData conferences for Yellowbrick maintainers and helps us buy promotional material such as stickers and t-shirts.

`NumFOCUS`_: Yellowbrick is a NumFOCUS affiliated project (not a fiscally sponsored project). Our relationship with NumFOCUS has given us a lot of data science cred in the community by being listed on their website. We are also eligible to apply for small development grants and infrastructure support. We often participate in the project developers mailing list and other activities such as Google Summer of Code.

`Georgetown University`_: Georgetown primarily provides space for Yellowbrick events including the research labs. Additionally, Georgetown Data Science Certificate students are introduced to Yellowbrick at the beginning of their machine learning education and we often perform user testing of new features on them!

How to Support Yellowbrick
~~~~~~~~~~~~~~~~~~~~~~~~~~

Yellowbrick is developed by volunteers who work on the project in their spare time and not as part of their regular full-time work. If Yellowbrick has become critical to the success of your organization, please consider giving back to Yellowbrick.

    "... open source thrives on human rather than financial resources. There
    are many ways to grow human resources, such as distributing the
    workload among more contributors or encouraging companies to
    make open source part of their employees’ work. An effective
    support strategy must include multiple ways to generate time and
    resources besides directly financing development. It must start from
    the principle that the open source approach is not inherently flawed,
    but rather under-resourced."

    -- `Roads and Bridges: The Unseen Labor Behind our Digital Infrastructure <https://www.fordfoundation.org/about/library/reports-and-studies/roads-and-bridges-the-unseen-labor-behind-our-digital-infrastructure/>`_

The main thing that the Yellowbrick maintainers need is *time*. There are many ways to provide that time through non-financial mechanisms such as:

- Create a written policy in your company handbook that dedicates time for your employees to contribute to open source projects like Yellowbrick.
- Interact with our community giving encouragement and advice, particularly for long term planning and non-code related activities like design and documentation.
- Advocate and evangelize your use of Yellowbrick and other open source software through blog posts and social media.
- Consider long term support strategies rather than ad hoc or one-off actions.
- Teach your students Machine Learning with Yellowbrick.

More concrete and financial support is also welcome, particularly if it's directed through a specific effort. If you are interested in this kind of support consider:

- Making a donation to NumFOCUS on behalf of Yellowbrick.
- Engaging District Data Labs for coporate training on visual machine learning with Yellowbrick (which will directly support Yellowbrick maintainers).
- Supporting your employee's continuing professional education in the Georgetown Data Science Certificate.
- Providing long term support for fixed costs such as hosting.

Yellowbrick's mission is to enhance the machine learning workflow through open source visual steering and diagnostics. If you're interested in a more formal affiliate relationship to support this mission, please get in contact with us directly.

License
-------

Yellowbrick is an open source project and its `license <https://github.com/DistrictDataLabs/yellowbrick/blob/master/LICENSE.txt>`_ is an implementation of the FOSS `Apache 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_ license by the Apache Software Foundation. `In plain English <https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)>`_ this means that you can use Yellowbrick for commercial purposes, modify and distribute the source code, and even sublicense it. We want you to use Yellowbrick, profit from it, and contribute back if you do cool things with it.

There are, however, a couple of requirements that we ask from you. First, when you copy or distribute Yellowbrick source code, please include our copyright and license found in the `LICENSE.txt <https://github.com/DistrictDataLabs/yellowbrick/blob/master/LICENSE.txt>`_ at the root of our software repository. In addition, if we create a file called "NOTICE" in our project you must also include that in your source distribution. The "NOTICE" file will include attribution and thanks to those who have worked so hard on the project! Note that you may not use our names, trademarks, or logos to promote your work or in any other way than to reference Yellowbrick. Finally, we provide Yellowbrick with no warranty and you can't hold any Yellowbrick contributor or affiliate liable for your use of our software.

We think that's a pretty fair deal, and we're big believers in open source. If you make any changes to our software, use it commercially or academically, or have any other interest, we'd love to hear about it.

Presentations
-------------

Yellowbrick has enjoyed the spotlight in several presentations at recent conferences. We hope that these notebooks, talks, and slides will help you understand Yellowbrick a bit better.

Papers:
    - `Yellowbrick: Visualizing the Scikit-Learn Model Selection Process <http://joss.theoj.org/papers/10.21105/joss.01075>`_

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


Citing Yellowbrick
------------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1206239.svg
   :target: https://doi.org/10.5281/zenodo.1206239

.. image:: http://joss.theoj.org/papers/10.21105/joss.01075/status.svg
   :target: https://doi.org/10.21105/joss.01075

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
        version = {0.9.1},
        author = {Bengfort, Benjamin and Bilbro, Rebecca and Danielsen, Nathan and
            Gray, Larry and {McIntyre}, Kristen and Roman, Prema and Poh, Zijie and
            others},
        date = {2018-11-14},
        year = {2018},
        doi = {10.5281/zenodo.1206264}
    }

You can also find DOI (digital object identifiers) for every version of Yellowbrick on `zenodo.org <https://doi.org/10.5281/zenodo.1206239>`_; use the BibTeX on this site to reference specific versions or changes made to the software.

We've also published a paper in the `Journal of Open Source Software (JOSS) <http://joss.theoj.org/papers/10.21105/joss.01075>`_ that discusses how Yellowbrick is designed to influence the model selection workflow. You may cite this paper if you are discussing Yellowbrick more generally in your research (instead of a specific version) or are interested in discussing visual analytics or visualization for machine learning. Please cite *Bengfort and Bilbro (2019)* with the following BibTex:

.. code-block:: bibtex

    @article{bengfort_yellowbrick_2019,
        title = {Yellowbrick: {{Visualizing}} the {{Scikit}}-{{Learn Model Selection Process}}},
        journaltitle = {The Journal of Open Source Software},
        volume = {4},
        number = {35},
        series = {1075},
        date = {2019-03-24},
        year = {2019},
        author = {Bengfort, Benjamin and Bilbro, Rebecca},
        url = {http://joss.theoj.org/papers/10.21105/joss.01075},
        doi = {10.21105/joss.01075}
    }

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

.. _QuatroCinco: https://flic.kr/p/2Yj9mj
.. _API: http://scikit-learn.org/stable/modules/classes.html
.. _SIGMOD: http://cseweb.ucsd.edu/~arunkk/vision/SIGMODRecord15.pdf
.. _Wikipedia: https://en.wikipedia.org/wiki/Yellow_brick_road
.. _`@rebeccabilbro`: https://github.com/rebeccabilbro
.. _`@bbengfort`: https://github.com/bbengfort
.. _`District Data Labs`: http://www.districtdatalabs.com/
.. _`Georgetown University`: https://scs.georgetown.edu/programs/375/certificate-in-data-science/
.. _`NumFOCUS`: https://numfocus.org/