.. -*- mode: rst -*-

Mushroom
========

From Audobon Society Field Guide; mushrooms described in terms of physical characteristics; classification: poisonous or edible.

=================   ==============================
Samples total                                 8124
Dimensionality                 4 (reduced from 22)
Features                               categorical
Targets               str: {"edible", "poisonous"}
Task(s)                             classification
=================   ==============================

Description
-----------

This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be" for Poisonous Oak and Ivy.

Citation
--------

Downloaded from the `UCI Machine Learning Repository <https://archive.ics.uci.edu/ml/datasets/Mushroom>`_  on February 28, 2017.

Schlimmer, Jeffrey Curtis. "Concept acquisition through representational adjustment." (1987).

Langley, Pat. "Trading off simplicity and coverage in incremental concept learning." Machine Learning Proceedings 1988 (2014): 73.

Duch, Włodzisław, Rafał Adamczak, and Krzysztof Grabczewski. "Extraction of logical rules from training data using backpropagation networks." The 1st Online Workshop on Soft Computing. 1996.

Duch, Wlodzislaw, Rafal Adamczak, and Krzysztof Grabczewski. "Extraction of crisp logical rules using constrained backpropagation networks." (1997).

Loader
------

.. autofunction:: yellowbrick.datasets.loaders.load_mushroom
