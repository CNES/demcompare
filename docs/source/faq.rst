.. _faq:

.. role:: bash(code)
   :language: bash


Frenquently Asked Questions
===========================

Installation troubleshootings
*****************************

Depending on pip version, installation problems can happen with packages dependencies installation order. Install and upgrade pip and numpy if demcompare installation crashes:

.. code-block:: bash

    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade numpy

.. note:: Be careful: Rasterio has its own embedded version of GDAL. Please use rasterio no-binary version in Makefile install if you want to use a GDAL local version:

    .. code-block:: bash

        python3 -m pip install --no-binary rasterio rasterio

Step by step run troubleshootings
*********************************

.. note::  Be careful: the positional argument for the configuration file can be wrongly considered as a step if used after the :bash:`--step` option.

    .. code-block:: bash

        demcompare --step stats config.json : KO
        demcompare config.json --step stats : OK

