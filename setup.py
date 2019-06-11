from setuptools import setup
from subprocess import check_output
from codecs import open


requirements = ['numpy',
                'scipy',
                'pygdal=={}.*'.format(check_output(['gdal-config', '--version']).rstrip().decode("utf-8")),
                'pyproj',
                'matplotlib',
                'astropy',
                'sphinx']


def readme():
    with open('readme.md', "r", "utf-8") as f:
        return f.read()


setup(name='dem_compare',
      version='0.1',
      description='A tool to compare Digital Elevation Models',
      long_description=readme(),
      url='https://framagit.org/jmichel-otb/dem_compare',
      author='CNES',
      author_email='julien.michel@cnes.fr',
      license='GNU LGPLv3',
      packages=['dem_compare'],
      scripts=['cli/cli-dem_compare.py', 'cli/compare_with_baseline.py'],
      install_requires=requirements)
