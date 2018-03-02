#!/usr/bin/env python
# setup
# Setup script for installing yellowbrick
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Wed May 18 14:33:26 2016 -0400
#
# Copyright (C) 2016 District Data Labs
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
NAME         = "yellowbrick"
DESCRIPTION  = "A suite of visual analysis and diagnostic tools for machine learning."
AUTHOR       = "Rebecca Bilbro, Benjamin Bengfort"
EMAIL        = "info@districtdatalabs.com"
MAINTAINER   = "Benjamin Bengfort"
LICENSE      = "Apache 2"
REPOSITORY   = "https://github.com/districtdatalabs/yellowbrick"
PACKAGE      = "yellowbrick"

## Define the keywords
KEYWORDS     = ('visualization', 'machine learning', 'scikit-learn', 'matplotlib', 'data science')

## Define the classifiers
## See https://pypi.python.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS  = (
    'Development Status :: 3 - Alpha',
    'Environment :: Other Environment',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Topic :: Software Development',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Scientific/Engineering :: Visualization',
)

## Important Paths
PROJECT      = os.path.abspath(os.path.dirname(__file__))
REQUIRE_PATH = "requirements.txt"
VERSION_PATH = os.path.join(PACKAGE, "version.py")
PKG_DESCRIBE = "DESCRIPTION.rst"

## Directories to ignore in find_packages
EXCLUDES     = (
    "tests", "bin", "docs", "fixtures", "register", "notebooks", "examples",
)

##########################################################################
## Helper Functions
##########################################################################

def read(*parts):
    """
    Assume UTF-8 encoding and return the contents of the file located at the
    absolute path from the REPOSITORY joined with *parts.
    """
    with codecs.open(os.path.join(PROJECT, *parts), 'rb', 'utf-8') as f:
        return f.read()


def get_version(path=VERSION_PATH):
    """
    Reads the __init__.py defined in the VERSION_PATH to find the get_version
    function, and executes it to ensure that it is loaded correctly.
    """
    namespace = {}
    exec(read(path), namespace)
    return namespace['get_version'](short=True)


def get_requires(path=REQUIRE_PATH):
    """
    Yields a generator of requirements as defined by the REQUIRE_PATH which
    should point to a requirements.txt output by `pip freeze`.
    """
    for line in read(path).splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            yield line

##########################################################################
## Define the configuration
##########################################################################

config = {
    "name": NAME,
    "version": get_version(),
    "description": DESCRIPTION,
    "long_description": read(PKG_DESCRIBE),
    "license": LICENSE,
    "author": AUTHOR,
    "author_email": EMAIL,
    "maintainer": MAINTAINER,
    "maintainer_email": EMAIL,
    "url": REPOSITORY,
    "download_url": "{}/tarball/v{}".format(REPOSITORY, get_version()),
    "packages": find_packages(where=PROJECT, exclude=EXCLUDES),
    "install_requires": list(get_requires()),
    "classifiers": CLASSIFIERS,
    "keywords": KEYWORDS,
    "zip_safe": False,
    "scripts": [],
    "setup_requires":["pytest-runner"],
    "tests_require":["pytest"],
}


##########################################################################
## Run setup script
##########################################################################

if __name__ == '__main__':
    setup(**config)
