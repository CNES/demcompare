.. _dem_tools_modules:

Dem Tools modules
=================

This section describes all dem tools modules: :ref:`dem_tools`, :ref:`dataset_tools`, :ref:`img_tools`.

As explained below, the `dem_tools`_ module handles the main API for dem manipulation through a dataset described in `dataset_tools`_ below.

.. _dem_tools:

Dem_tools module
----------------

**demcompare.dem_tools** module file is `dem_tools.py <https://github.com/CNES/demcompare/blob/master/demcompare/dem_tools.py>`_

This module contains main functions to manipulate DEM raster images.

It represents the primary API to manipulate DEM as xarray dataset in demcompare.
Dataset and associated internal functions are described in `dataset_tools`_

As one can see in :ref:`demcompare_module`, the main demcompare module in `__init__.py <https://github.com/CNES/demcompare/blob/master/demcompare/__init__.py>`_ file uses `dem_tools`_'s
functions such as **load_dem**, **reproject_dems** and **compute_alti_diff_for_stats**.

The full list of API functions available in the `dem_tools`_ module, as well as their description and
input and output parameters can be found here: :doc:`/api_reference/demcompare/dem_tools/index`

.. _dataset_tools:

Dataset_tools module
--------------------

**demcompare.dataset_tools** module file is `dataset_tools.py <https://github.com/CNES/demcompare/blob/master/demcompare/dataset_tools.py>`_

This module contains functions associated to demcompare's DEM dataset creation. It shall not be used directly,
as it is the `dem_tools`_ module who handles its API.

The **demcompare DEM dataset** is a xarray Dataset created by demcompare for each DEM, it contains all the necessary information
for demcompare to perform all the different available processes on a DEM.

One can see here all the information inside a demcompare dataset:

.. _demcompare_dataset:

.. code-block:: text

    :image: 2D (row, col) image as xarray.DataArray,
    :georef_transform: 1D (trans_len) xarray.DataArray with the parameters:

                - c: x-coordinate of the upper left pixel,
                - a: pixel size in the x-direction in map units/pixel,
                - b: rotation about x-axis,
                - f: y-coordinate of the upper left pixel,
                - d: rotation about y-axis,
                - e: pixel size in the y-direction in map units, negative

    :classification_layers: 3D (row, col, indicator) xarray.DataArray:

                It contains the maps of all classification layers,
                being the indicator a list with each
                classification_layer name.

    :attributes:

                - nodata : image nodata value. float
                - input_img : image input path. str or None
                - crs : image crs. rasterio.crs.CRS
                - xres : x resolution (value of transform[1]). float
                - yres : y resolution (value of transform[5]). float
                - plani_unit : georefence planimetric unit. astropy.units
                - zunit : input image z unit value. astropy.units
                - bounds : image bounds. rasterio.coords.BoundingBox
                - geoid_path : geoid path. str or None
                - source_rasterio : rasterio's DatasetReader object or None.


.. _img_tools:

Img_tools module
----------------

**demcompare.img_tools** module file is `img_tools.py <https://github.com/CNES/demcompare/blob/master/demcompare/img_tools.py>`_

This module contains generic functions associated to raster images.
It consists mainly on wrappers to rasterio functions. Like `dataset_tools`_, this module shall not be used directly,
as it is the `dem_tools`_ module who handles its API.
