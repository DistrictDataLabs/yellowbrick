.. _examples/yellowbrick-palettes:

=========
Palettes
=========

Yellowbrick includes custom palettes as well as familiar ones from Matplotlib and Seaborn.

.. code:: python

    %matplotlib inline

    import matplotlib.pyplot as plt
    from yellowbrick.style.palettes import PALETTES, color_palette

.. code:: python

    # ['blue', 'green', 'red', 'maroon', 'yellow', 'cyan']
    for palette in PALETTES.keys():
        color_palette(palette).plot()
        plt.title(palette, loc='left')



.. image:: images/palettes_2_0.png



.. image:: images/palettes_2_1.png



.. image:: images/palettes_2_2.png



.. image:: images/palettes_2_3.png



.. image:: images/palettes_2_4.png



.. image:: images/palettes_2_5.png



.. image:: images/palettes_2_6.png



.. image:: images/palettes_2_7.png



.. image:: images/palettes_2_8.png



.. image:: images/palettes_2_9.png



.. image:: images/palettes_2_10.png



.. image:: images/palettes_2_11.png



.. image:: images/palettes_2_12.png



.. image:: images/palettes_2_13.png



.. image:: images/palettes_2_14.png



.. image:: images/palettes_2_15.png



.. image:: images/palettes_2_16.png



.. image:: images/palettes_2_17.png
