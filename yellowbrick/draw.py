# yellowbrick.draw
# Utilities for common matplotlib drawing procedures.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Sun Aug 19 10:35:50 2018 -0400
#
# ID: draw.py [] benjamin@bengfort.com $

"""
Utilities for common matplotlib drawing procedures.
"""

##########################################################################
## Imports
##########################################################################

from .base import Visualizer
from .exceptions import YellowbrickValueError
from .style.colors import resolve_colors

from matplotlib import patches

import matplotlib.pyplot as plt
import numpy as np

##########################################################################
## Legend Drawing Utilities
##########################################################################

def manual_legend(g, labels, colors, **legend_kwargs):
    """
    Adds a manual legend for a scatter plot to the visualizer where the labels
    and associated colors are drawn with circle patches instead of determining
    them from the labels of the artist objects on the axes. This helper is
    used either when there are a lot of duplicate labels, no labeled artists,
    or when the color of the legend doesn't exactly match the color in the
    figure (e.g. because of the use of transparency).

    Parameters
    ----------
    g : Visualizer or Axes object
        The graph to draw the legend on, either a Visualizer or a matplotlib
        Axes object. If None, the current axes are drawn on, but this is not
        recommended.

    labels : list of str
        The text labels to associate with the legend. Note that the labels
        will be added to the legend in the order specified.

    colors : list of colors
        A list of any valid matplotlib color reference. The number of colors
        specified must be equal to the number of labels.

    legend_kwargs : dict
        Any additional keyword arguments to pass to the legend.

    Returns
    -------
    legend: Legend artist
        The artist created by the ax.legend() call, returned for further
        manipulation if required by the caller.

    Notes
    -----
    Right now this method simply draws the patches as rectangles and cannot
    take into account the line or scatter plot properties (e.g. line style or
    marker style). It is possible to add Line2D patches to the artist that do
    add manual styles like this, which we can explore in the future.

    .. seealso:: https://matplotlib.org/gallery/text_labels_and_annotations/custom_legends.html
    """
    # Get access to the matplotlib Axes
    if isinstance(g, Visualizer):
        g = g.ax
    elif g is None:
        g = plt.gca()

    # Ensure that labels and colors are the same length to prevent odd behavior.
    if len(colors) != len(labels):
        raise YellowbrickValueError(
            "please specify the same number of colors as labels!"
        )

    # Create the legend handles with the associated colors and labels
    handles = [
        patches.Patch(color=color, label=label)
        for color, label in zip(colors, labels)
    ]

    # Return the Legend artist
    return g.legend(handles=handles, **legend_kwargs)


def bar_stack(data, ax=None, labels=None, ticks=None, colors=None, 
              orientation='vertical', colormap=None, **kwargs):
    """
    An advanced bar chart plotting utility that can draw bar and stacked bar charts from
    data, wrapping calls to the specified matplotlib.Axes object. 

    Parameters
    ----------
    data : 2D array-like 
        The data passed to the Visualizer. Rows represent each stack in the bar chart and columns
        represent each bar. Therefore, a single bar chart is created by passing a 2D array
        containing a single row, while the data to create a bar chart with 3 stacks would have a
        shape of (3, b).

    ax : matplotlib.Axes, default: None
        The axes object to draw the barplot on, uses plt.gca() if not specified.

    labels : list of str, default: None
        The labels for each row in the bar stack, used to create a legend. 

    ticks : list of str, default: None
        The labels for each bar, added to the x-axis for a vertical plot, or the y-axis
        for a horizontal plot.  

    colors : array-like, default: None
        Specify the colors of each bar, each row in the stack, or every segment.

    colormap : string or matplotlib cmap
        Specify a colormap for each bar, each row in the stack, or every segment.

    kwargs : dict 
        Additional keyword arguments to pass to ``ax.bar``. 
    """
    if ax is None:
        ax = plt.gca()
    
    colors = resolve_colors(n_colors=data.shape[0], 
                            colormap=colormap, 
                            colors=colors)    
    
    idx = np.arange(data.shape[1])
    prev = np.zeros(data.shape[1])
    orientation = orientation.lower()
    if orientation.startswith('v'):
        for rdx,row in enumerate(data):
            ax.bar(idx, 
                   row,
                   bottom = prev,
                   color = colors[rdx])
            prev+=row
        ax.set_xticks(idx)
        if ticks is not None:
            ax.set_xticklabels(ticks, rotation=90)

    elif orientation.startswith('h'):
        for rdx,row in enumerate(data):
            ax.barh(idx, 
                   row,
                   left = prev,
                   color = colors[rdx])
            prev+=row
        ax.set_yticks(idx)
        if ticks is not None:
            ax.set_yticklabels(ticks)        
    else:
        raise YellowbrickValueError(
                "unknown orientation '{}'".format(orientation)
                )
    
    # Generates default labels is labels are not specified.
    labels = labels or np.arange(data.shape[0])

    manual_legend(ax, labels=labels, colors=colors)
    return ax
