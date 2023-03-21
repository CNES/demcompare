Demcompare's modules
====================

The following modules are part of demcompare's architecture: 


- **__init__.py**

This module includes demcompare's run function, which performs the input cfg's steps.

- **dem_tools.py**

This module contains main functions to manipulate DEM raster images.

It represents the primary API to manipulate DEM as xarray dataset in demcompare.
Dataset and associated internal functions are described in dataset_tools.py

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

- **demcompare.py**

This module includes demcompare's main and input parser.

- **helpers_init.py**

In this module high level parameters are checked and default options are set. Some helper functions to handle
the output paths from the __init__ are also included here.

- **log_conf.py**

The logconf module in demcompare contains logging configuration functions.

- **output_tree_design.py**

Module containing the default output tree design architecture for demcompare's output directory.

- **report.py**

Module in charge of generating output demcompare's report to visualize the results (graphs, stats, ...)

- **sphinx_project_generator.py**

sphinx_project_generator is a module containing the helper functions for the creation of the output demcompare's report
and ease its manipulation.
