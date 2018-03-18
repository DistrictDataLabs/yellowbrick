# tests.sklearn_test_utils
# Helper functions and utils for testing visualizers, adapted from sklearn
#
# Author:   Nathan Danielsen <nathan.danielsen@gmail.com>
# Created:  Sun Oct 09 12:23:13 2016 -0400
#
# Copyright (C) 2016 District Data Labs
# For license information, see LICENSE.txt
#
# ID: base.py [b8e3318] nathan.danielsen@gmail.com $


import sys
import warnings

import numpy as np

def clean_warning_registry():
    """Safe way to reset warnings."""
    warnings.resetwarnings()
    reg = "__warningregistry__"
    for mod_name, mod in list(sys.modules.items()):
        if 'six.moves' in mod_name:
            continue
        if hasattr(mod, reg):
            getattr(mod, reg).clear()


def assert_warns_message(warning_class, message, func, *args, **kw):
    # very important to avoid uncontrolled state propagation
    """Test that a certain warning occurs and with a certain message.
    Parameters. (From sklearn.utils.testing)
    ----------
    warning_class : the warning class
        The class to test for, e.g. UserWarning.
    message : str | callable
        The message or a substring of the message to test for. If callable,
        it takes a string as the argument and will trigger an AssertionError
        if the callable returns `False`.
    func : callable
        Callable object to trigger warnings.
    *args : the positional arguments to `func`.
    **kw : the keyword arguments to `func`.
    Returns
    -------
    result : the return value of `func`
    """
    clean_warning_registry()
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        if hasattr(np, 'VisibleDeprecationWarning'):
            # Let's not catch the numpy internal DeprecationWarnings
            warnings.simplefilter('ignore', np.VisibleDeprecationWarning)
        # Trigger a warning.
        result = func(*args, **kw)
        # Verify some things
        if not len(w) > 0:
            raise AssertionError("No warning raised when calling %s"
                                 % func.__name__)

        found = [issubclass(warning.category, warning_class) for warning in w]
        if not any(found):
            raise AssertionError("No warning raised for %s with class "
                                 "%s"
                                 % (func.__name__, warning_class))

        message_found = False
        # Checks the message of all warnings belong to warning_class
        for index in [i for i, x in enumerate(found) if x]:
            # substring will match, the entire message with typo won't
            msg = w[index].message  # For Python 3 compatibility
            msg = str(msg.args[0] if hasattr(msg, 'args') else msg)
            if callable(message):  # add support for certain tests
                check_in_message = message
            else:
                check_in_message = lambda msg: message in msg

            if check_in_message(msg):
                message_found = True
                break

        if not message_found:
            raise AssertionError("Did not receive the message you expected "
                                 "('%s') for <%s>, got: '%s'"
                                 % (message, func.__name__, msg))

    return result
