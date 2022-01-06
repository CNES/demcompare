.. _faq:

Frenquently Asked Questions
===========================

Troubleshootings
****************

Depending on pip version, installation problems can happen with packages dependencies installation order. Install and upgrade pip, numpy and cython if demcompare installation crashes:

.. code-block:: bash

    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade numpy cython

.. note:: Be careful: Rasterio has its own embedded version of GDAL. Please use rasterio no-binary version in Makefile install if you want to use a GDAL local version:

    .. code-block:: bash

        python3 -m pip install --no-binary rasterio rasterio


