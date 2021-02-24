#!/usr/bin/env python
# coding: utf8
# Install demcompare, whether via
#      ``python setup.py install``
#    or
#      ``pip install demcompare``
"""
Packaging setup.py for compatibility
All packaging in setup.cfg, except setuptools_scm version
"""

import setuptools

try:
    setuptools.setup(use_scm_version=True)
except:
    print(
        "\n\nAn error occurred while building the project, "
        "please ensure you have the most updated version of setuptools, "
        "setuptools_scm and wheel with:\n"
        "   pip install -U setuptools setuptools_scm wheel\n\n"
    )
    raise
