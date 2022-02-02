.. _coregistration:

Inside **demcompare**'s code, the two input DEMs are refered to as **ref** (reference DEM) and **sec** and (secondary DEM).
For this reason, from now on we will refer to each DEM as **ref** and **sec** respectively.

Coregistration
==============

To perform the **DEMs comparison**, **demcompare** considers that the **ref** is the cleaner DEM and has higher resolution than **sec**. Considering this, if the DEMs need to be coregistered, **demcompare** will perform the following steps:

- The **ref** is reprojected to the **sec**'s **georeference grid**, so that it's resolution will be the same as **sec**'s. The **ref** is the one to be adapted as it supposedly is the cleaner one.
- The [NuthKaab]_ corregistration algorithm performs the corregistration between both DEMs by interpolating and resampling the **ref**, so that it will compute two coregistred DEMs that have the **sec**'s **georeference grid** and **georeference origin**.
- Since in **demcompare** the **ref**'s georeference origin is considered the correct location, both coregistred DEM's are then translated to the **ref**'s **georeference origin** by a simple transform using the offsets obtained by Nuth et Kaab.
- With both coregistred DEMs having the **sec**'s **georeference grid** and the **ref**'s **georeference origin**, **demcompare** is ready to compare both DEMs computing a wide variety of standard metrics and statistics.


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


For the Nuth & Kääb universal coregistration algorithm :

.. [NuthKaab] Nuth, C. Kääb, 2011. A. Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change. Cryosphere 5, 271290.
