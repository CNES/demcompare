.. _disparity:

Coregistration
==============

Inside Demcompare's code, the two input DEMs are refered to as ref (reference DEM) and dem and (secondary DEM).
If the DEMs are uncoregistered, DEMcompare will perform the following steps:

The ref is reprojected to the dem's georeference grid, so that it's resolution will be the same as dem's. The ref is the one to be adapted as it supposedly is the cleaner one.
The Nuth et Kaab corregistration algorithm performs the corregistration between both DEMs by interpolating and resampling the ref, so that it will compute two coregistred DEMs that have the dem's georeference grid and georeference origin.
Since in DEMcompare the ref's georeference origin is considered the correct location, both coregistred DEM's are then translated to the ref's georeference origin by a simple transform using the offsets obtained by Nuth et Kaab.
With both coregistred DEMs having the dem's georeference grid and the ref's georeference origin, DEMcompare is ready to compare both DEMs computing a wide variety of standard metrics and statistics.


WGS84 and geoid references
**************************

The DEMs altitudes can rely on both ellipsoid and geoid references.
However one shall use the georef parameter to set
the reference assigned to the DEMs (the two DEMs can rely on different references).
The default geoid reference is EGM96. However, the user can set another geoid using the parameter
geoid_path.

Altimetric unit
***************

It is assumed both DEMs altimetric unit is meter.
If otherwise, one shall use the zunit to set the actual altimetric
unit.

ROI definition
**************
The processed Region of interest (ROI) is either defined by either the image coordinates (x,y) of its top-left corner,
and its dimensions (w, h) in pixels as a python dictionary with 'x', 'y', 'w' and 'h' keys or the geographical
coordinates of the projected image as tuple with (left, bottom, right, top) coordinates

In anyway, this is four numbers that ought to be given in the json configuration file.
The ROI refers to the tested DEM and will be adapted to the REF dem georef by demcompare itself.
If no ROI definition is provided then DEMs raster are fully processed.

Tile processing
***************
Tile processing is not available anymore. A future version might provide a better way to deal with very large data. In
the meantime one can deal with heavy DSMs by setting a ROI (see previous chapter).