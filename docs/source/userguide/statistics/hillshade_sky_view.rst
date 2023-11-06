.. _hillshade_sky_view:

Hillshade and sky-view factor representations
=============================================

A DEM is usually represented with a map of its elevation, but this view can sometimes be challenging to interpret. 
Therefore, more visual representations were added into demcompare to help analyse a DEM: Hillshade and sky-view factor representations. 
These additional visualizations are illustrated in the figure below.

The hillshade is a representation based on slope orientation. 
By convention, a north-oriented slope is assigned a value of 0°, an east-oriented slope 90°, a south-oriented slope 180°, and a west-oriented slope 270°. 
As a result, east-oriented slopes appear darker than west-oriented ones in this visualization.

This hillshade visualization not only helps users grasp the topography or 3D aspect of a DEM but also helps in identifying artifacts or other anomalous features.

On the other hand, the sky view factor assumes that the light is arriving from all the directions instead of coming from a punctual luminous source (the sun). 
Then, a point is lit up proportionally to the solid angle of sky visible from its position. 
This makes it particularly easy to pinpoint features such as valleys. 
This representation serves as a valuable complement to the hillshade.

.. figure:: /images/HillshadeSVF.png
    :width: 1000px
    :align: center

    Hillshade (left) and sky-view factor (right) for a DEM located in the Pyrenees.