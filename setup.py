#!/usr/bin/env python
# setup
# Setup script for installing yellowbrick
#
# Author:   Benjamin Bengfort
# Created:  Wed May 18 14:33:26 2016 -0400
#
# Copyright (C) 2016 The scikit-yb developers
# For license information, see LICENSE.txt and NOTICE.md
#
# ID: setup.py [c4f3ba7] benjamin@bengfort.com $

"""
Setup script for installing yellowbrick.
See http://bbengfort.github.io/programmer/2016/01/20/packaging-with-pypi.html
"""

##########################################################################
## Imports
##########################################################################

import os
import codecs

from setuptools import setup
from setuptools import find_packages

##########################################################################
## Package Information
##########################################################################

## Basic information
## Basic information
NAME         = "yellowbrick"
DESCRIPTION  = "A suite of visual analysis and diagnostic tools for machine learning."
AUTHOR       = "The scikit-yb developers"
EMAIL        = "yellowbrick@googlegroups.com"
MAINTAINER   = "The scikit-yb developers"
LICENSE      = "Apache 2"
REPOSITORY   = "https://github.com/DistrictDataLabs/yellowbrick"
PACKAGE      = "yellowbrick"
URL          = "http://scikit-yb.org/"

## Define the keywords
KEYWORDS = (
    "visualization",
    "machine learning",
    "scikit-learn",
    "matplotlib",
    "data science",
)

## Define the classifiers
## See https://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS  = (
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering :: Visualization",
)

## Important Paths
PROJECT = os.path.abspath(os.path.dirname(__file__))
REQUIRE_PATH = "requirements.txt"
VERSION_PATH = os.path.join(PACKAGE, "version.py")
PKG_DESCRIBE = "DESCRIPTION.md"

## Directories to ignore in find_packages
EXCLUDES = (
    "tests", "tests.*",
    "bin",
    "docs", "docs.*",
    "fixtures",
    "register",
    "notebooks", "notebooks.*",
    "examples", "examples.*",
    "binder", "binder.*",
    "paper",
)

##########################################################################
## Helper Functions
##########################################################################


def read(*parts):
    """
    Assume UTF-8 encoding and return the contents of the file located at the
    absolute path from the REPOSITORY joined with *parts.
    """
    with codecs.open(os.path.join(PROJECT, *parts), "rb", "utf-8") as f:
        return f.read()


def get_version(path=VERSION_PATH):
    """
    Reads the python file defined in the VERSION_PATH to find the get_version
    function, and executes it to ensure that it is loaded correctly. Separating
    the version in this way ensures no additional code is executed.
    """
    namespace = {}
    exec(read(path), namespace)
    return namespace["get_version"](short=True)


def get_requires(path=REQUIRE_PATH):
    """
    Yields a generator of requirements as defined by the REQUIRE_PATH which
    should point to a requirements.txt output by `pip freeze`.
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            yield line


def get_description_type(path=PKG_DESCRIBE):
    """
    Returns the long_description_content_type based on the extension of the
    package describe path (e.g. .txt, .rst, or .md).
    """
    _, ext = os.path.splitext(path)
    return {".rst": "text/x-rst", ".txt": "text/plain", ".md": "text/markdown"}[ext]


##########################################################################
## Define the configuration
##########################################################################

config = {
    "name": NAME,
    "version": get_version(),
    "description": DESCRIPTION,
    "long_description": read(PKG_DESCRIBE),
    "long_description_content_type": get_description_type(PKG_DESCRIBE),
    "classifiers": CLASSIFIERS,
    "keywords": KEYWORDS,
    "license": LICENSE,
    "author": AUTHOR,
    "author_email": EMAIL,
    "url": URL,
    "maintainer": MAINTAINER,
    "maintainer_email": EMAIL,
    "project_urls": {
        "Documentation": URL,
        "Download": "{}/tarball/v{}".format(REPOSITORY, get_version()),
        "Source": REPOSITORY,
        "Tracker": "{}/issues".format(REPOSITORY),
    },
    "download_url": "{}/tarball/v{}".format(REPOSITORY, get_version()),
    "packages": find_packages(where=PROJECT, exclude=EXCLUDES),
    "package_data": {"yellowbrick": ["datasets/manifest.json"]},
    "zip_safe": False,
    "entry_points": {"console_scripts": []},
    "install_requires": list(get_requires()),
    "python_requires": ">=3.4, <4",
    "setup_requires": ["pytest-runner"],
    "tests_require": ["pytest"],
}


##########################################################################
## Run setup script
##########################################################################

if __name__ == "__main__":
    setup(**config)
