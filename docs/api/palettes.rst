.. -*- mode: rst -*-

Colors and Style
================

Yellowbrick believes that visual diagnostics are more effective if visualizations are appealing. As a result, we have borrowed familiar styles from `Seaborn <http://seaborn.pydata.org/tutorial/aesthetics.html>`_ and use the new `Matplotlib 2.0 styles <https://matplotlib.org/users/colormaps.html>`_. We hope that these out of the box styles will make your visualizations publication ready, though of course you can customize your own look and feel by directly modifying the visualization with matplotlib.

Yellowbrick prioritizes color in its visualizations for most visualizers. There are two types of color sets that can be provided to a visualizer: a palette and a sequence. Palettes are discrete color values usually of a fixed length and are typically used for classification or clustering by showing each class, cluster or topic. Sequences are continuous color values that do not have a fixed length but rather a range and are typically used for regression or clustering, showing all possible values in the target or distances between items in clusters.

In order to make the distinction easy, most matplotlib colors (both palettes and sequences) can be referred to by name. A complete listing can be imported as follows:

.. code:: python

    import matplotlib.pyplot as plt
    from yellowbrick.style.palettes import PALETTES, SEQUENCES, color_palette

Palettes and sequences can be passed to visualizers as follows:

.. code:: python

    visualizer = Visualizer(color="bold")

Refer to the API listing of each visualizer for specifications about how each color argument is handled. In the next two sections we will show every possible color palette and sequence currently available in Yellowbrick.

Color Palettes
--------------

Color palettes are discrete color lists that have a fixed length. The most common palettes are ordered as "blue", "green", "red", "maroon", "yellow", "cyan", and an optional "key". This allows you to specify these named colors or by the first character, e.g. 'bgrmyck' for matplotlib visualizations.

To change the global color palette, use the `set_palette` function as follows:

.. code:: python

    from yellowbrick.style import set_palette
    set_palette('flatui')

Color palettes are most often used for classifiers to show the relationship between discrete class labels. They can also be used for clustering algorithms to show membership in discrete clusters.

A complete listing of the Yellowbrick color palettes can be visualized as follows:

.. code:: python

    # ['blue', 'green', 'red', 'maroon', 'yellow', 'cyan']
    for palette in PALETTES.keys():
        color_palette(palette).plot()
        plt.title(palette, loc='left')



.. image:: images/palettes/palettes_2_0.png



.. image:: images/palettes/palettes_2_1.png



.. image:: images/palettes/palettes_2_2.png



.. image:: images/palettes/palettes_2_3.png



.. image:: images/palettes/palettes_2_4.png



.. image:: images/palettes/palettes_2_5.png



.. image:: images/palettes/palettes_2_6.png



.. image:: images/palettes/palettes_2_7.png



.. image:: images/palettes/palettes_2_8.png



.. image:: images/palettes/palettes_2_9.png



.. image:: images/palettes/palettes_2_10.png



.. image:: images/palettes/palettes_2_11.png



.. image:: images/palettes/palettes_2_12.png



.. image:: images/palettes/palettes_2_13.png



.. image:: images/palettes/palettes_2_14.png



.. image:: images/palettes/palettes_2_15.png



.. image:: images/palettes/palettes_2_16.png


Color Sequences
---------------

Color sequences are continuous representations of color and are usually defined as a fixed number of steps between a minimum and maximal value. Sequences must be created with a total number of bins (or length) before plotting to ensure that values are assigned correctly. In the listing below, each sequence is shown with varying lengths to describe the range of colors in detail.

Color sequences are most often used in regressions to show the distribution in the range of target values. They can also be used in clustering and distribution analysis to show distance or histogram data.

Below is a complete listing of all the sequence names available in Yellowbrick:

.. code:: python

    for name, maps in SEQUENCES.items():
        for num, palette in maps.items():
            color_palette(palette).plot()
            plt.title("{} - {}".format(name, num), loc='left')


.. image:: images/palettes/palettes_3_1.png



.. image:: images/palettes/palettes_3_2.png



.. image:: images/palettes/palettes_3_3.png



.. image:: images/palettes/palettes_3_4.png



.. image:: images/palettes/palettes_3_5.png



.. image:: images/palettes/palettes_3_6.png



.. image:: images/palettes/palettes_3_7.png



.. image:: images/palettes/palettes_3_8.png



.. image:: images/palettes/palettes_3_9.png



.. image:: images/palettes/palettes_3_10.png



.. image:: images/palettes/palettes_3_11.png



.. image:: images/palettes/palettes_3_12.png



.. image:: images/palettes/palettes_3_13.png



.. image:: images/palettes/palettes_3_14.png



.. image:: images/palettes/palettes_3_15.png



.. image:: images/palettes/palettes_3_16.png



.. image:: images/palettes/palettes_3_17.png



.. image:: images/palettes/palettes_3_18.png



.. image:: images/palettes/palettes_3_19.png



.. image:: images/palettes/palettes_3_20.png



.. image:: images/palettes/palettes_3_21.png



.. image:: images/palettes/palettes_3_22.png



.. image:: images/palettes/palettes_3_23.png



.. image:: images/palettes/palettes_3_24.png



.. image:: images/palettes/palettes_3_25.png



.. image:: images/palettes/palettes_3_26.png



.. image:: images/palettes/palettes_3_27.png



.. image:: images/palettes/palettes_3_28.png



.. image:: images/palettes/palettes_3_29.png



.. image:: images/palettes/palettes_3_30.png



.. image:: images/palettes/palettes_3_31.png



.. image:: images/palettes/palettes_3_32.png



.. image:: images/palettes/palettes_3_33.png



.. image:: images/palettes/palettes_3_34.png



.. image:: images/palettes/palettes_3_35.png



.. image:: images/palettes/palettes_3_36.png



.. image:: images/palettes/palettes_3_37.png



.. image:: images/palettes/palettes_3_38.png



.. image:: images/palettes/palettes_3_39.png



.. image:: images/palettes/palettes_3_40.png



.. image:: images/palettes/palettes_3_41.png



.. image:: images/palettes/palettes_3_42.png



.. image:: images/palettes/palettes_3_43.png



.. image:: images/palettes/palettes_3_44.png



.. image:: images/palettes/palettes_3_45.png



.. image:: images/palettes/palettes_3_46.png



.. image:: images/palettes/palettes_3_47.png



.. image:: images/palettes/palettes_3_48.png



.. image:: images/palettes/palettes_3_49.png



.. image:: images/palettes/palettes_3_50.png



.. image:: images/palettes/palettes_3_51.png



.. image:: images/palettes/palettes_3_52.png



.. image:: images/palettes/palettes_3_53.png



.. image:: images/palettes/palettes_3_54.png



.. image:: images/palettes/palettes_3_55.png



.. image:: images/palettes/palettes_3_56.png



.. image:: images/palettes/palettes_3_57.png



.. image:: images/palettes/palettes_3_58.png



.. image:: images/palettes/palettes_3_59.png



.. image:: images/palettes/palettes_3_60.png



.. image:: images/palettes/palettes_3_61.png



.. image:: images/palettes/palettes_3_62.png



.. image:: images/palettes/palettes_3_63.png



.. image:: images/palettes/palettes_3_64.png



.. image:: images/palettes/palettes_3_65.png



.. image:: images/palettes/palettes_3_66.png



.. image:: images/palettes/palettes_3_67.png



.. image:: images/palettes/palettes_3_68.png



.. image:: images/palettes/palettes_3_69.png



.. image:: images/palettes/palettes_3_70.png



.. image:: images/palettes/palettes_3_71.png



.. image:: images/palettes/palettes_3_72.png



.. image:: images/palettes/palettes_3_73.png



.. image:: images/palettes/palettes_3_74.png



.. image:: images/palettes/palettes_3_75.png



.. image:: images/palettes/palettes_3_76.png



.. image:: images/palettes/palettes_3_77.png



.. image:: images/palettes/palettes_3_78.png



.. image:: images/palettes/palettes_3_79.png



.. image:: images/palettes/palettes_3_80.png



.. image:: images/palettes/palettes_3_81.png



.. image:: images/palettes/palettes_3_82.png



.. image:: images/palettes/palettes_3_83.png



.. image:: images/palettes/palettes_3_84.png



.. image:: images/palettes/palettes_3_85.png



.. image:: images/palettes/palettes_3_86.png



.. image:: images/palettes/palettes_3_87.png



.. image:: images/palettes/palettes_3_88.png



.. image:: images/palettes/palettes_3_89.png



.. image:: images/palettes/palettes_3_90.png



.. image:: images/palettes/palettes_3_91.png



.. image:: images/palettes/palettes_3_92.png



.. image:: images/palettes/palettes_3_93.png



.. image:: images/palettes/palettes_3_94.png



.. image:: images/palettes/palettes_3_95.png



.. image:: images/palettes/palettes_3_96.png



.. image:: images/palettes/palettes_3_97.png



.. image:: images/palettes/palettes_3_98.png



.. image:: images/palettes/palettes_3_99.png



.. image:: images/palettes/palettes_3_100.png



.. image:: images/palettes/palettes_3_101.png



.. image:: images/palettes/palettes_3_102.png



.. image:: images/palettes/palettes_3_103.png



.. image:: images/palettes/palettes_3_104.png



.. image:: images/palettes/palettes_3_105.png



.. image:: images/palettes/palettes_3_106.png



.. image:: images/palettes/palettes_3_107.png



.. image:: images/palettes/palettes_3_108.png



.. image:: images/palettes/palettes_3_109.png



.. image:: images/palettes/palettes_3_110.png



.. image:: images/palettes/palettes_3_111.png



.. image:: images/palettes/palettes_3_112.png



.. image:: images/palettes/palettes_3_113.png



.. image:: images/palettes/palettes_3_114.png



.. image:: images/palettes/palettes_3_115.png



.. image:: images/palettes/palettes_3_116.png



.. image:: images/palettes/palettes_3_117.png



.. image:: images/palettes/palettes_3_118.png



.. image:: images/palettes/palettes_3_119.png



.. image:: images/palettes/palettes_3_120.png



.. image:: images/palettes/palettes_3_121.png



.. image:: images/palettes/palettes_3_122.png



.. image:: images/palettes/palettes_3_123.png



.. image:: images/palettes/palettes_3_124.png



.. image:: images/palettes/palettes_3_125.png



.. image:: images/palettes/palettes_3_126.png



.. image:: images/palettes/palettes_3_127.png



.. image:: images/palettes/palettes_3_128.png



.. image:: images/palettes/palettes_3_129.png



.. image:: images/palettes/palettes_3_130.png



.. image:: images/palettes/palettes_3_131.png



.. image:: images/palettes/palettes_3_132.png



.. image:: images/palettes/palettes_3_133.png



.. image:: images/palettes/palettes_3_134.png



.. image:: images/palettes/palettes_3_135.png



.. image:: images/palettes/palettes_3_136.png



.. image:: images/palettes/palettes_3_137.png



.. image:: images/palettes/palettes_3_138.png



.. image:: images/palettes/palettes_3_139.png



.. image:: images/palettes/palettes_3_140.png



.. image:: images/palettes/palettes_3_141.png



.. image:: images/palettes/palettes_3_142.png



.. image:: images/palettes/palettes_3_143.png



.. image:: images/palettes/palettes_3_144.png



.. image:: images/palettes/palettes_3_145.png



.. image:: images/palettes/palettes_3_146.png



.. image:: images/palettes/palettes_3_147.png



.. image:: images/palettes/palettes_3_148.png



.. image:: images/palettes/palettes_3_149.png



.. image:: images/palettes/palettes_3_150.png



.. image:: images/palettes/palettes_3_151.png



.. image:: images/palettes/palettes_3_152.png



.. image:: images/palettes/palettes_3_153.png



.. image:: images/palettes/palettes_3_154.png



.. image:: images/palettes/palettes_3_155.png



.. image:: images/palettes/palettes_3_156.png



.. image:: images/palettes/palettes_3_157.png



.. image:: images/palettes/palettes_3_158.png



.. image:: images/palettes/palettes_3_159.png



.. image:: images/palettes/palettes_3_160.png



.. image:: images/palettes/palettes_3_161.png



.. image:: images/palettes/palettes_3_162.png



.. image:: images/palettes/palettes_3_163.png



.. image:: images/palettes/palettes_3_164.png



.. image:: images/palettes/palettes_3_165.png



.. image:: images/palettes/palettes_3_166.png



.. image:: images/palettes/palettes_3_167.png



.. image:: images/palettes/palettes_3_168.png



.. image:: images/palettes/palettes_3_169.png



.. image:: images/palettes/palettes_3_170.png



.. image:: images/palettes/palettes_3_171.png



.. image:: images/palettes/palettes_3_172.png



.. image:: images/palettes/palettes_3_173.png



.. image:: images/palettes/palettes_3_174.png



.. image:: images/palettes/palettes_3_175.png



.. image:: images/palettes/palettes_3_176.png



.. image:: images/palettes/palettes_3_177.png



.. image:: images/palettes/palettes_3_178.png



.. image:: images/palettes/palettes_3_179.png



.. image:: images/palettes/palettes_3_180.png



.. image:: images/palettes/palettes_3_181.png



.. image:: images/palettes/palettes_3_182.png



.. image:: images/palettes/palettes_3_183.png



.. image:: images/palettes/palettes_3_184.png



.. image:: images/palettes/palettes_3_185.png



.. image:: images/palettes/palettes_3_186.png



.. image:: images/palettes/palettes_3_187.png



.. image:: images/palettes/palettes_3_188.png



.. image:: images/palettes/palettes_3_189.png



.. image:: images/palettes/palettes_3_190.png



.. image:: images/palettes/palettes_3_191.png



.. image:: images/palettes/palettes_3_192.png



.. image:: images/palettes/palettes_3_193.png



.. image:: images/palettes/palettes_3_194.png



.. image:: images/palettes/palettes_3_195.png



.. image:: images/palettes/palettes_3_196.png



.. image:: images/palettes/palettes_3_197.png



.. image:: images/palettes/palettes_3_198.png



.. image:: images/palettes/palettes_3_199.png



.. image:: images/palettes/palettes_3_200.png



.. image:: images/palettes/palettes_3_201.png



.. image:: images/palettes/palettes_3_202.png



.. image:: images/palettes/palettes_3_203.png



.. image:: images/palettes/palettes_3_204.png



.. image:: images/palettes/palettes_3_205.png



.. image:: images/palettes/palettes_3_206.png



.. image:: images/palettes/palettes_3_207.png



.. image:: images/palettes/palettes_3_208.png

API Reference
-------------

yellowbrick.style.colors module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: yellowbrick.style.colors
    :members:
    :undoc-members:
    :show-inheritance:

yellowbrick.style.palettes module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: yellowbrick.style.palettes
    :members:
    :undoc-members:
    :show-inheritance:

yellowbrick.style.rcmod module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: yellowbrick.style.rcmod
    :members:
    :undoc-members:
    :show-inheritance:
