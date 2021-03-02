#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of demcompare
# (see https://github.com/CNES/demcompare).
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
# flake8: noqa: E501
"""
sphinx_project_generator was built for dsm_compare.py and its 'report' step.
As so, it is designed for this simple purpose, meaning only light version of
the Makefile and the conf.py are proposed, and no way to control them.
If somehow required, this project generator shall be improved to offer some
level of customization.
"""
# Standard imports
import errno
import os
import shutil
import subprocess


def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class SphinxProjectManager:
    """
    Sphinx Projet Manager Class
    Class to represent the Sphinx demcompare report
    and be able to ease its manipulation:
    create a sphinx project architecture + configuration
    + build and install project
    """

    def __init__(self, working_dir, output_dir, index_name, project_name):
        self._working_dir = working_dir
        self._output_dir = output_dir
        self._index_name = index_name
        self._project_name = project_name

        self._makefile = None
        self._confpy = None

        self._create_sphinx_project()

    def _create_sphinx_project(self):
        """
        Create project architecture and configuration :

        _working_dir/build/
                   /Makefile
                   /source/
                           conf.py
                           _index_name.rst

        :return:
        """

        # Create directories
        self._build_dir = os.path.join(self._working_dir, "build")
        mkdir_p(self._build_dir)
        self._src_dir = os.path.join(self._working_dir, "source")
        mkdir_p(self._src_dir)

        # Create configuration files
        self._makefile = os.path.join(self._working_dir, "Makefile")
        self._create_makefile()
        self._confpy = os.path.join(self._src_dir, "conf.py")
        self._create_confpy()

        # clean old doc
        cur_dir = os.curdir
        try:
            os.chdir(self._working_dir)
            subprocess.check_call(
                ["make", "clean"], stderr=subprocess.STDOUT, env=os.environ
            )
            os.chdir(cur_dir)
        except Exception:
            os.chdir(cur_dir)
            raise
        else:
            print("Sphinx clean succeeded ")

    def _create_makefile(self):
        # pylint: disable=line-too-long, anomalous-backslash-in-string
        if self._makefile:
            make_contents = "\n".join(
                [
                    "# Makefile for Sphinx documentation",
                    "SPHINXOPTS    =",
                    "SPHINXBUILD   = sphinx-build",
                    "PAPER         =",
                    "BUILDDIR      = build",
                    "",
                    "# User-friendly check for sphinx-build",
                    "ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)",
                    "$(error The sphinx-build command was not found. Make sure you have Sphinx installed.)",
                    "endif",
                    "",
                    "# Internal variables.",
                    "PAPEROPT_a4     = -D latex_paper_size=a4",
                    "PAPEROPT_letter = -D latex_paper_size=letter",
                    "ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source",
                    "# the i18n builder cannot share the environment and doctrees with the others",
                    "I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source",
                    "",
                    ".PHONY: clean",
                    "clean:",
                    "\trm -rf $(BUILDDIR)/*",
                    "",
                    ".PHONY: html",
                    "html:",
                    "\t$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html",
                    "\t@echo",
                    '\t@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."',
                    "",
                    ".PHONY: latex",
                    "latex:",
                    "\t$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex",
                    "\t@echo",
                    '\t@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."',
                    '\t@echo "Run \`make\' in that directory to run these through (pdf)latex" \\',
                    '\t"(use \`make latexpdf\' here to do that automatically)."',
                    "",
                    ".PHONY: latexpdf",
                    "latexpdf:",
                    "\t$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex",
                    '\t@echo "Running LaTeX files through pdflatex..."',
                    "\t$(MAKE) -C $(BUILDDIR)/latex all-pdf",
                    '\t@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."',
                    "",
                    ".PHONY: doctest",
                    "doctest:",
                    "\t$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest",
                    '\t@echo "Testing of doctests in the sources finished, look at the " \\',
                    '              "results in $(BUILDDIR)/doctest/output.txt."',
                ]
            )

            with open(self._makefile, "w") as makefile:
                makefile.write(make_contents)

    def _create_confpy(self):
        # pylint: disable=line-too-long
        if self._confpy:
            conf_contents = "\n".join(
                [
                    "# -*- coding: utf-8 -*-",
                    "",
                    "# -- General configuration ------------------------------------------------",
                    "numfig=True",
                    "source_suffix = '.rst'",
                    "source_encoding = 'utf-8'",
                    "master_doc = '{}'".format(self._index_name),
                    "project = u'{}'".format(self._project_name),
                    "copyright = u'2017, CS, CNES'",
                    "author = u'CS'",
                    "language = 'en'",
                    "pygments_style = 'sphinx'",
                    "# -- Options for HTML output ----------------------------------------------",
                    "#html_logo = 'Images/cs_cnes_200pixels.bmp'",
                    "html_show_copyright = False",
                    "html_search_language = 'en'",
                    "htmlhelp_basename = 'dsm_compare_report'",
                    "",
                    "# -- Options for LaTeX output ---------------------------------------------",
                    "latex_documents = [(master_doc, '{}.tex', u'{}', u'CS', 'manual')]".format(
                        self._index_name, self._project_name
                    ),
                    "#latex_logo = None" "",
                    "# -- Options for docx output ---------------------------------------------",
                    "docx_template = 'template.docx'",
                    "templates_path = ['_templates']",
                ]
            )

            with open(self._confpy, "w") as confpy:
                confpy.write(conf_contents)

    def write_body(self, body):
        with open(
            os.path.join(self._src_dir, self._index_name + ".rst"), "w"
        ) as rst_file:
            rst_file.write(body)

    def build_project(self, mode):
        """

        :param mode: 'html' or 'latex' or 'latexpdf'
        :return:
        """

        cur_dir = os.curdir
        try:
            os.chdir(self._working_dir)
            cr_build = open(
                os.path.join(self._working_dir, "cr_build-{}.txt".format(mode)),
                "w",
            )
            subprocess.check_call(
                ["make", mode],
                stdout=cr_build,
                stderr=subprocess.STDOUT,
                env=os.environ,
            )
            os.chdir(cur_dir)
        except:
            os.chdir(cur_dir)
            raise
        else:
            print(("Sphinx build succeeded for {} mode".format(mode)))

    def install_project(self):
        """
        Method to install (copy from build) sphinx project in _output_dir
        """
        # Remove previous tree (and ignore errors)
        shutil.rmtree(self._output_dir, ignore_errors=True)
        # Copy build directory to install directory
        shutil.copytree(self._build_dir, self._output_dir)
        print(
            (
                "Documentation installed under {} directory".format(
                    self._output_dir
                )
            )
        )

    @staticmethod
    def clean():
        cmd = "make clean"
        os.system(cmd)


if __name__ == "__main__":
    SPM = SphinxProjectManager(".", ".", "test", "Projet de Test")
    SPM.write_body(
        "\n".join(
            [
                ".. _ProjetTest",
                "",
                "*************",
                " Projet Test",
                "*************",
                "",
                "Presentation du projet Test",
            ]
        )
    )
    SPM.build_project("html")
    SPM.install_project()
