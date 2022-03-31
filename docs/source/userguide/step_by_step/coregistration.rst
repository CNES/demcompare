.. _coregistration:

Inside **demcompare**'s code, the two input DEMs are refered to as **ref** (reference DEM) and **sec** and (secondary DEM).
For this reason, from now on we will refer to each DEM as **ref** and **sec** respectively.

Coregistration
==============

Demcompare performs the DEMs coregistration with [NuthKaab]_ corregistration algorithm.

In order to apply the coregistration offsets to the **sec** so that it will have the same georeference origin as the **ref**, the following GDAL command may be used:

.. code-block:: bash

    gdal_translate -a_ullr <ulx> <uly> <lrx> <lry> /PATH_TO/secondary_dem.tif /PATH_TO/coreg_secondary_dem.tif

Being *<ulx> <uly> <lrx> <lry>* the coordinate bounds of the offsets applied on **sec**. They are stored in the *final_cfg.json* file as **gdal_translate_bounds**.

The coregistration step also computes the *coregDEM.tif* and *coregREF.tif*, which are the intermediate coregistered DEMs used for the stats computation.

.. note:: Please notice that *coregDEM.tif* and *coregREF.tif* share the same georeference origin, **but this origin may not be the reference one**. Moreover, those DEMs have been reprojected and cropped in order to have the same resolution and size.


WGS84 and geoid references
**************************

The DEMs altitudes can rely on both **ellipsoid** and **geoid** references.
However one shall use the `georef` parameter to set the reference assigned to the DEMs (the two DEMs can rely on different references).
The default geoid reference is EGM96. However, the user can set another geoid using the parameter `geoid_path`.

For more information about the input DEMs reference, please refer to :ref:`inputs`

Altimetric unit
***************

It is assumed both DEMs altimetric unit is **meter**.
If otherwise, one shall use the *zunit* to set the actual altimetric unit.

ROI definition
**************
The processed Region of interest (:term:`ROI`) is either defined by:

- The image coordinates *(x,y)* of its top-left corner, and its dimensions (w, h) in pixels as a python dictionary with `x`, `y`, `w` and `h` keys.
- The geographical coordinates of the **projected image** as tuple with *(left, bottom, right, top)* coordinates. For instance, for a DSM whose Coordinate Reference System is **EPSG:32630**, a possible ROI would be *(left=600255.0, bottom=4990745.0, right=709255.0, top=5099745.0)*.

In anyway, this is four numbers that ought to be given in the `json` configuration file.

The ROI refers to the **sec** and will be adapted to the **ref** georeference origin by **demcompare** itself.

If no ROI definition is provided then DEMs raster are fully processed.

Tile processing
***************
Tile processing is not available anymore. A future version might provide a better way to deal with very large data. In
the meantime one can deal with heavy DSMs by setting a :term:`ROI` (see previous chapter).

References
**********

For the Nuth & Kääb universal coregistration algorithm :

.. [NuthKaab] Nuth, C. Kääb, 2011. A. Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change. Cryosphere 5, 271290.
