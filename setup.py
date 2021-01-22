# Install dem_compare, whether via
#      ``python setup.py install``
#    or
#      ``pip install dem_compare``
"""
This file is the dem_compare package main program.
"""

from setuptools import setup
from subprocess import check_output
from codecs import open as copen

# Meta-data.
NAME = 'dem_compare'
DESCRIPTION = 'A tool to compare Digital Elevation Models'
URL = 'https://github.com/CNES/dem_compare'
AUTHOR = 'CNES'
REQUIRES_PYTHON = '>=3.6.0'
EMAIL = 'julien.michel@cnes.fr'
LICENSE = 'Apache License 2.0'
REQUIREMENTS = ['numpy',
                'xarray>=0.13.*',
                'scipy',
                'rasterio',
                'pyproj',
                'matplotlib',
                'astropy',
                'sphinx',
                'lib_programname']

def readme():
    with copen('readme.md', "r", "utf-8") as fstream:
        return fstream.read()

setup(
    name=NAME,
    use_scm_version=True,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    packages=['dem_compare'],
    include_package_data = True,
    long_description=readme(),
    install_requires=REQUIREMENTS,
    python_requires=REQUIRES_PYTHON,
    scripts=['cli/cli-dem_compare.py', 'cli/compare_with_baseline.py']
)
