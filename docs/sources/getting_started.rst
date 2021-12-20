Getting started
===============

Overview
########

DEMcompare is a python software that aims at comparing two DEMs together.
A DEM is a 3D computer graphics representation of elevation data to represent terrain.
DEMcompare has several characteristics:

- Provides a wide variety of standard metrics and allows one to classify the statistics.
- Works whether or not the two DEMs share common format, projection system,
- Planimetric resolution, and altimetric unit.
- The coregistration algorithm is based on the Nuth & Kääb universal coregistration method.
- The default behavior classifies the stats by slope ranges but one can provide any other data to classify the stats.
A comparison report can be compiled as html or pdf documentation with statistics printed as tables and plots.

Install
#######

Only Linux Plaforms are supported (virtualenv or bare machine) with Python 3 installed.
The default install mode is via the pip package, typically through a virtualenv:
.. code-block:: bash
    python3 -m venv venv
    source venv/bin/activate
    pip install demcompare

Troubleshootings
Depending on pip version, installation problems can happen with packages dependencies installation order. Install and upgrade pip, numpy and cython if demcompare installation crashes:
.. code-block:: bash
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade numpy cython
Be careful: Rasterio have its own embedded version of GDAL
Please use rasterio no-binary version in Makefile install if you want to use a GDAL local version:
.. code-block:: bash
    python3 -m pip install --no-binary rasterio rasterio


First step
##########

Run the python script demcompare with a json configuration file as unique
argument (see tests/test_config.json as an example):
.. code-block:: bash
    cd tests/
    demcompare test_config.json
The results can be observed with:
.. code-block:: bash
    firefox test_output/doc/published_report/html/demcompare_report.html &
demcompare can be launched with a file containing its parameters (one per line) with "@" character:
.. code-block:: bash
demcompare @opts.txt
opts.txt example file:
test_config.json
--display

Customize
#########

Credits
#######

Related
#######

* `CARS <https://github.com/CNES/CARS>`_ - CNES 3D reconstruction software

References
##########

For more details about the NMAD metric :
Höhle, J., Höhle, M., 2009. Accuracy assessment of Digital Elevation Models by means of robust statistical methods.
ISPRS Journal of Photogrammetry and Remote Sensing 64(4), 398-406.
For the Nuth & Kääb universal coregistration algorithm :
Nuth, C. Kääb, 2011. A. Co-registration and bias corrections of satellite elevation data sets for quantifying glacier
thickness change. Cryosphere 5, 271290.