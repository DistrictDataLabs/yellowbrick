# yellowbrick.draw
# Utilities for common matplotlib drawing procedures.
#
# Author:  Benjamin Bengfort
# Created: Sun Aug 19 10:35:50 2018 -0400
#
# Copyright (C) 2018 The sckit-yb developers
# For license information, see LICENSE.txt
#
# ID: draw.py [dd915ad] benjamin@bengfort.com $

"""
Utilities for common matplotlib drawing procedures.
"""

##########################################################################
## Imports
##########################################################################

from .base import Visualizer
from .exceptions import YellowbrickValueError
from .style.colors import resolve_colors

from matplotlib import axes, patches, lines

import matplotlib.pyplot as plt
import numpy as np

##########################################################################
## Legend Drawing Utilities
##########################################################################

def manual_legend(g, labels, colors=None, styles=None, **legend_kwargs):
    """
    Adds a manual legend for a scatter plot to the visualizer. The legend
    entries are drawn according to the ``styles`` parameter if specified, and
    with circle patches (colored according to ``colors``) if not specified.
    Calling this function overrides the default behavior of drawing the legend 
    from the labels of the artist objects on the axes. 

    This helper is used either when there are a lot of duplicate labels, 
    no labeled artists, or when the color of the legend doesn't exactly 
    match the color in the figure (e.g. because of the use of transparency).

    Parameters
    ----------
    g : Visualizer or Axes object
        The graph to draw the legend on, either a Visualizer or a matplotlib
        Axes object. If None, the current axes are drawn on, but this is not
        recommended.

    labels : list of str
        The text labels to associate with the legend. Note that the labels
        will be added to the legend in the order specified.

    colors : list of colors, default: None
        A list of any valid matplotlib color references. If ``styles``
        is provided, colors must be either ``None`` or a list of equal length to
        ``labels``; in the latter case, this parameter takes predence over any 
        colors specified in ``styles``. To skip specifying a color for a
        particular entry, use an empty string, None, or 'None'.
        
    styles : list of str, default: None
        A list of matplotlib-style format strings, each corresponding to a label 
        and describing its graphical appearance in the legend, e.g., 'ro' for a 
        red circle. The number of styles specified must be equal to the number 
        of labels. Either one or both of ``colors`` and ``styles`` must be
        specified. Consistent with matplotlib, blank style entries default to 
        solid, unmarked, black lines.
        
    legend_kwargs : dict
        Any additional keyword arguments to pass to the legend.

    Returns
    -------
    legend: Legend artist
        The artist created by the ax.legend() call, returned for further
        manipulation if required by the caller.

    .. seealso:: https://matplotlib.org/gallery/text_labels_and_annotations/custom_legends.html

    .. seealso:: https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.plot.html
    """

    # Get access to the matplotlib Axes
    if isinstance(g, Visualizer):
        g = g.ax
    elif g is None:
        g = plt.gca()

    if styles:
        # Documented the `styles` parameter as being a list when really
        # it makes sense to accept it as a list or a tuple
        if type(styles) not in (list, tuple):
            raise YellowbrickValueError(
                "Please specify the styles parameter as a list of strings!"
            )

        if len(styles) != len(labels):
            raise YellowbrickValueError(
                "Please specify the styles parameter as a list of length "
                "equal to the number of labels!"
            )            

        if colors is not None and len(colors) != len(labels):
            raise YellowbrickValueError(
                "Please specify the colors parameter either as colors=None or "
                "a list of length equal to the number of labels. You can use "
                "an empty string or None as a placeholder for colors that "
                "are already specified in the corresponding styles entry."
            )
    else:
        if colors is None or len(colors) != len(labels):
            raise YellowbrickValueError(
                "Please specify the colors parameter as a list of length equal "
                "to the number of labels!"
            )

    # Set legend's artist handles to:
    #   linestyles/markers/colors specified by `styles` if passed in, or
    #   patches according to `colors` if it is not
    if styles:
        if colors is None:
            colors = [None] * len(styles)
        else:
            colors = [None if color in ("", " ", None, 'None') else color 
                for color in colors]

        handles = list()
        for style, color, label in zip(styles, colors, labels):
            linestyle, marker, style_color = \
                axes._base._process_plot_format(style)
            
            # colors parameter should take precedence over styles,
            #   consistent with matplotlib
            color = color or style_color or 'black'
            # _process_plot_format() above will have already set linestyle to 
            #   '-' and marker to 'None' if they weren't specified

            line_2d = lines.Line2D([0], [0], linestyle=linestyle, marker=marker, 
                color=color, label=label)
            handles.append(line_2d)
    else: 
        handles = [
            patches.Patch(color=color, label=label) for 
                color, label in zip(colors, labels)
            ]

    # Return the Legend artist
    return g.legend(handles=handles, **legend_kwargs)

def bar_stack(
    data,
    ax=None,
    labels=None,
    ticks=None,
    colors=None,
    colormap=None,
    orientation="vertical",
    legend=True,
    legend_kws=None,
    **kwargs
):
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

    orientation:‘vertical’ or ‘horizontal’
        Specifies a horizontal or vertical bar chart.

    legend : boolean, default: True
        If True, the function add a legend with the plot

    legend_kws : dict, default: None
        Additional keyword arguments for the legend components.

    kwargs : dict
        Additional keyword arguments to pass to ``ax.bar``.
    """
    if ax is None:
        ax = plt.gca()

    colors = resolve_colors(n_colors=data.shape[0], colormap=colormap, colors=colors)

    idx = np.arange(data.shape[1])
    zeros = np.zeros(data.shape[1])
    # Stores stacks for both side of plotting axes
    stack_arr = np.zeros((data.shape[1], 2))
    orientation = orientation.lower()

    if orientation.startswith("h"):

        for rdx in range(len(data)):
            stack = [stack_arr[j, int(data[rdx][j] > 0)] for j in range(len(data[rdx]))]
            ax.barh(idx, data[rdx], left=stack, color=colors[rdx])
            # Updates the stack for negative side of y-axis
            stack_arr[:, 0] += np.minimum(data[rdx], zeros)
            # Updates stack for positive side of y-axis
            stack_arr[:, 1] += np.maximum(data[rdx], zeros)
        ax.set_yticks(idx)
        if ticks is not None:
            ax.set_yticklabels(ticks)

    elif orientation.startswith("v"):
        for rdx in range(len(data)):
            stack = [stack_arr[j, int(data[rdx][j] > 0)] for j in range(len(data[rdx]))]
            ax.bar(idx, data[rdx], bottom=stack, color=colors[rdx])
            # Updates the stack for negative side of x-axis
            stack_arr[:, 0] += np.minimum(data[rdx], zeros)
            # Updates the stack for negative side of x-axis
            stack_arr[:, 1] += np.maximum(data[rdx], zeros)
        ax.set_xticks(idx)
        if ticks is not None:
            ax.set_xticklabels(ticks, rotation=90)

    else:
        raise YellowbrickValueError("unknown orientation '{}'".format(orientation))

    # Generates default labels is labels are not specified.
    labels = labels or np.arange(data.shape[0])

    if legend:
        legend_kws = legend_kws or {}
        manual_legend(ax, labels=labels, colors=colors, **legend_kws)
    return ax
    