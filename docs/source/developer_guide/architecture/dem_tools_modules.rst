.. _dem_tools_modules:

Dem Tools modules
=================


This section describes all dem tools modules: **dem_tools.py**, **dataset_tools.py** and **img_tools.py**.
As explained below, it is the `dem_tools.py` module who handles the API.


- **dem_tools.py**

This module contains main functions to manipulate DEM raster images.

It represents the primary API to manipulate DEM as xarray dataset in demcompare.
Dataset and associated internal functions are described in dataset_tools.py

As one can see in :ref:`demcompare_module`, the main demcompare module in `__init__.py` file uses `dem_tools`'s
functions such as `load_dem`, `reproject_dems` and `compute_alti_diff_for_stats`.

- **dataset_tools.py**

This module contains functions associated to demcompare's DEM dataset creation. It shall not be used directly,
as the it is the `dem_tools.py` module who handles its API.

The **demcompare dataset** is an xarray Dataset containing:

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

- **img_tools.py**

This module contains generic functions associated to raster images.
It consists mainly on wrappers to rasterio functions. Like `dataset_tools.py`, this module shall not be used directly,
as the it is the `dem_tools.py` module who handles its API.
