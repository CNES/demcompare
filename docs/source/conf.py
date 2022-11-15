# pylint: skip-file
# flake8: noqa
# coding: utf8
# Copyright (c) 2020 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of demcompare
#
#     https://github.com/CNES/demcompare
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from pkg_resources import get_distribution

sys.path.insert(0, os.path.abspath(""))

# -- Project information -----------------------------------------------------

project = "Demcompare"
copyright = "2020, CNES"
author = "CNES"
# The full version, including alpha/beta/rc tags

try:
    version = get_distribution("demcompare").version
    release = version
except Exception as error:
    print("WARNING: cannot find demcompare version")
    version = "Unknown"
    release = version

# The master toctree document.
master_doc = "index"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.imgmath",
    "autoapi.extension",
    'sphinx_tabs.tabs'
]

autoapi_dirs = ["../../demcompare"]
autoapi_root = "api_reference"
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
# Add any paths that contain templates here, relative to this directory.cd
templates_path = ["_templates"]

# These folders are copied to the documentation's HTML output
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ["css/my_custom.css"]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# Title
html_title = "Demcompare Documentation"
html_short_title = "Demcompare Documentation"

# Logo
html_logo = "images/demcompare_picto.png"

# Favicon
html_favicon = "images/favicon.ico"

# Theme options
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 3,
}

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

numfig = True
