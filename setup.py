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

import pkg_resources
import setuptools

pkg_resources.require('setuptools>=42')
setuptools.setup()
