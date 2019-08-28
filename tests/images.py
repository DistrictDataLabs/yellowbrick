# tests.images
# Helper utility to manage baseline images for image comparisons.
#
# Author:  Benjamin Bengfort <benjamin@bengfort.com>
# Created: Fri Mar 02 21:51:56 2018 -0500
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt
#
# ID: images.py [ab37b18] benjamin@bengfort.com $

"""
Helper utility to manage baseline images for image comparisons. Usage:

    $ python -m tests.images --help

The utility uses argparse to manage command line input for specific baseline
image movement or clearing.
"""

##########################################################################
## Imports
##########################################################################

import os
import glob
import shutil
import argparse


BASE = os.path.dirname(__file__)
BASELINE = os.path.join(BASE, "baseline_images")
ACTUAL = os.path.join(BASE, "actual_images")


##########################################################################
## Helper Methods
##########################################################################


def relpath(path):
    """
    Compute the path relative to the test directory.
    """
    path = path.rstrip(".py")
    path = os.path.relpath(path, start=BASE)
    if path.startswith("..") or path.startswith(os.path.sep):
        raise ValueError("path must include test directory")
    return path


def validate(path):
    """
    Validates a test directory path by checking if any actual images exist
    and creating the mirror baseline directory if required. Returns bool to
    show if work should be done on the path or raises an exception if invalid.
    """
    actual_path = os.path.join(ACTUAL, path)
    if not os.path.exists(actual_path):
        return False

    if not os.path.isdir(actual_path):
        raise ValueError("{} is not a directory".format(actual_path))

    baseline_path = os.path.join(BASELINE, path)
    if not os.path.exists(baseline_path):
        os.makedirs(baseline_path)

    return True


def clear(path):
    """
    Clear .png files in the actual and baseline directories
    """
    for root in (BASELINE, ACTUAL):
        for img in glob.glob(os.path.join(root, path, "*.png")):
            os.remove(img)
            print("removed {}".format(os.path.relpath(img)))


def list_images(path):
    """
    List the associated files for the given test path
    """
    # Get contents of directories
    bases = set(glob.glob(os.path.join(BASELINE, path, "*.png")))
    actual = set(glob.glob(os.path.join(ACTUAL, path, "*.png")))
    diffs = set(glob.glob(os.path.join(ACTUAL, path, "*-failed-diff.png")))

    # Dedupe actual and diffs
    actual -= diffs

    # Get test file names without parent dirs
    names = set(map(os.path.basename, bases)) | set(map(os.path.basename, actual))

    # Build listing
    output = []

    for name in names:
        bname, bext = os.path.splitext(name)

        output.append(bname)
        output.append("-" * len(bname))

        # Handle base path
        base_path = os.path.join(BASELINE, path, name)
        if base_path in bases:
            output.append("  - {}".format(os.path.relpath(base_path)))
        else:
            output.append("  - no baseline image")

        # Handle actual path
        actual_path = os.path.join(ACTUAL, path, name)
        if actual_path in actual:
            output.append("  - {}".format(os.path.relpath(actual_path)))
        else:
            output.append("  - no actual image")

        # Handle diff path
        diff_path = os.path.join(ACTUAL, path, "{}-failed-diff{}".format(bname, bext))
        if diff_path in diffs:
            output.append("  - {}".format(os.path.relpath(diff_path)))

        # Add breathing room
        output.append("")

    return "\n".join(output)


def sync(path):
    """
    Move all non-diff images from actual to baseline
    """
    for fname in os.listdir(os.path.join(ACTUAL, path)):
        if fname.endswith("-diff.png"):
            continue

        if fname.endswith(".png"):
            src = os.path.join(ACTUAL, path, fname)
            dst = os.path.join(BASELINE, path, fname)
            shutil.copy2(src, dst)
            print("synced {}".format(os.path.relpath(src, start=ACTUAL)))


##########################################################################
## Main Method
##########################################################################


def main(args):
    # Get directories relative to test dir
    test_dirs = list(map(relpath, args.test_dirs))

    # Validate directories and filter empty ones
    test_dirs = filter(validate, test_dirs)

    # If list, simply list what is available and exit
    if args.list:
        output = [list_images(path) for path in test_dirs]
        print("\n".join(output))
        return

    # If clear, clear the baseline and actual directories
    if args.clear:
        for path in test_dirs:
            clear(path)
        return

    # Move images that aren't diffs from actual to baseline
    for path in test_dirs:
        sync(path)


if __name__ == "__main__":

    args = {
        ("-C", "--clear"): {
            "action": "store_true",
            "help": "clear actual, diffs, and baseline images for test",
        },
        ("-L", "--list"): {
            "action": "store_true",
            "help": "list images images for tests and exit",
        },
        "test_dirs": {
            "metavar": "DIR",
            "nargs": "+",
            "help": "directories to move images from actual to baseline",
        },
    }

    # Create the parser and add the arguments
    parser = argparse.ArgumentParser(
        description="utility to manage baseline images for comparisons",
        epilog="report any issues on GitHub",
    )

    for pargs, kwargs in args.items():
        if isinstance(pargs, str):
            pargs = (pargs,)
        parser.add_argument(*pargs, **kwargs)

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        parser.error(str(e))
