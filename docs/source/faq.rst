.. _faq:

.. role:: bash(code)
   :language: bash


Frequently Asked Questions
===========================

Installation troubleshooting
****************************

Depending on pip version, installation problems can happen with packages dependencies installation order. Install and upgrade pip and numpy if demcompare installation crashes:

.. code-block:: bash

    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade numpy

.. note:: Be careful: Rasterio has its own embedded version of GDAL. Please use rasterio no-binary version in Makefile install if you want to use a GDAL local version:

    .. code-block:: bash

        python3 -m pip install --no-binary rasterio rasterio

Data Dimension Management
*************************

There are no constraints on the dimensions of input data, except that they must share a geographical footprint (cf: Intersection DEM schema).
However, classification masks must have the same dimensions as the associated DEM.
The constraints and returned error we impose are directly derived from the rasterio mask class . `Documentation rasterio mask`_

.. figure:: /images/dem_intersection.png
    :width: 1000px
    :align: center

    Intersection DEM schema

.. _`Documentation rasterio mask`: https://rasterio.readthedocs.io/en/stable/api/rasterio.mask.html
