#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
sphinx_project_generator was built for dsm_compare.py and its 'report' step.
As so, it is designed for this simple purpose, meaning only light version of
the Makefile and the conf.py are proposed, and no way to control them.
If somehow required, this project generator shall be improved to offer some
level of customization.
"""

import os
import shutil
import errno
import subprocess


def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    try:
        os.makedirs(path)
    except OSError as exc: # requires Python > 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class SphinxProjectManager(object):
    def __init__(self, working_dir, output_dir, index_name, project_name):
        self._workingDir = working_dir
        self._outputDir = output_dir
        self._indexName = index_name
        self._projectName = project_name

        self._makefile = None
        self._confpy = None

        self._create_sphinx_project()

    def _create_sphinx_project(self):
        """
        Create project architecture and configuration :

        _workingDir/build/
                   /Makefile
                   /source/
                           conf.py
                           _indexName.rst

        :return:
        """

        # Create directories
        self._buildDir = os.path.join(self._workingDir, 'build')
        mkdir_p(self._buildDir)
        self._srcDir = os.path.join(self._workingDir, 'source')
        mkdir_p(self._srcDir)

        # Create configuration files
        self._makefile = os.path.join(self._workingDir, 'Makefile')
        self._create_makefile()
        self._confpy = os.path.join(self._srcDir, 'conf.py')
        self._create_confpy()

    def _create_makefile(self):
        if self._makefile:
            make_contents = '\n'.join(
                ['# Makefile for Sphinx documentation',
                 'SPHINXOPTS    =',
                 'SPHINXBUILD   = sphinx-build',
                 'PAPER         =',
                 'BUILDDIR      = build',
                 '',
                 '# User-friendly check for sphinx-build',
                 'ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)',
                 '$(error The sphinx-build command was not found. Make sure you have Sphinx installed.)',
                 'endif',
                 '',
                 '# Internal variables.',
                 'PAPEROPT_a4     = -D latex_paper_size=a4',
                 'PAPEROPT_letter = -D latex_paper_size=letter',
                 'ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source',
                 '# the i18n builder cannot share the environment and doctrees with the others',
                 'I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source',
                 '',
                 '.PHONY: clean',
                 'clean:',
                 '\trm -rf $(BUILDDIR)/*',
                 '',
                 '.PHONY: html',
                 'html:',
                 '\t$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html',
                 '\t@echo',
                 '\t@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."',
                 '',
                 '.PHONY: latex',
                 'latex:',
                 '\t$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex',
                 '\t@echo',
                 '\t@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."',
                 '\t@echo "Run \`make\' in that directory to run these through (pdf)latex" \\',
                 '\t"(use \`make latexpdf\' here to do that automatically)."',
                 '',
                 '.PHONY: latexpdf',
                 'latexpdf:',
                 '\t$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex',
                 '\t@echo "Running LaTeX files through pdflatex..."',
                 '\t$(MAKE) -C $(BUILDDIR)/latex all-pdf',
                 '\t@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."',
                 '',
                 '.PHONY: doctest',
                 'doctest:',
                 '\t$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest',
                 '\t@echo "Testing of doctests in the sources finished, look at the " \\',
                 '              "results in $(BUILDDIR)/doctest/output.txt."'
                 ]
            )

            with open(self._makefile, 'w') as makefile:
                makefile.write(make_contents)

    def _create_confpy(self):
        if self._confpy:
            conf_contents = '\n'.join(
                ['# -*- coding: utf-8 -*-',
                 '',
                 '# -- General configuration ------------------------------------------------',
                 'numfig=True',
                 'source_suffix = \'.rst\'',
                 'source_encoding = \'utf-8\'',
                 'master_doc = \'{}\''.format(self._indexName),
                 'project = u\'{}\''.format(self._projectName),
                 'copyright = u\'2017, CS, CNES\'',
                 'author = u\'CS\'',
                 'language = \'en\'',
                 'pygments_style = \'sphinx\'',
                 '# -- Options for HTML output ----------------------------------------------',
                 '#html_logo = \'Images/cs_cnes_200pixels.bmp\'',
                 'html_show_copyright = False',
                 'html_search_language = \'en\'',
                 'htmlhelp_basename = \'dsm_compare_report\'',
                 '',
                 '# -- Options for LaTeX output ---------------------------------------------',
                 'latex_documents = [(master_doc, \'{}.tex\', u\'{}\', u\'CS\', \'manual\')]'.format(self._indexName,
                                                                                                     self._projectName),
                 '#latex_logo = None'
                 '',
                 '# -- Options for docx output ---------------------------------------------',
                 'docx_template = \'template.docx\'',
                 'templates_path = [\'_templates\']'
                 ]
            )

            with open(self._confpy, 'w') as confpy:
                confpy.write(conf_contents)

    def write_body(self, body):
        with open(os.path.join(self._srcDir, self._indexName+'.rst'), 'w') as rst_file:
            rst_file.write(body)

    def build_project(self, mode):
        """

        :param mode: 'html' or 'latex' or 'latexpdf'
        :return:
        """

        cur_dir = os.curdir
        try:
            os.chdir(self._workingDir)
            cr_build = open(os.path.join(self._workingDir, 'cr_build-{}.txt'.format(mode)), 'w')
            subprocess.check_call(['make', mode], stdout=cr_build, stderr=subprocess.STDOUT, env=os.environ)
            os.chdir(cur_dir)
        except:
            os.chdir(cur_dir)
            raise
        else:
            print(('Sphinx build succeeded for {} mode'.format(mode)))

    def install_project(self):
        try:
            shutil.rmtree(self._outputDir)
        except:
            pass
        shutil.copytree(self._buildDir, self._outputDir)
        print(('Documentation installed under {} directory'.format(self._outputDir)))

    def clean(self):
        cmd = 'make clean'
        os.system(cmd)


if __name__ == '__main__':
    SPM = SphinxProjectManager('.', '.', 'test', 'Projet de Test')
    SPM.write_body('\n'.join([
        '.. _ProjetTest',
        '',
        '*************',
        ' Projet Test',
        '*************',
        '',
        'Presentation du projet Test'
    ]))
    SPM.build_project('html')
    SPM.install_project()
