.. _scientific_guide:

Scientific guide
================

.. _hillshade_sky_view:

Hillshade and sky-view factor representations
*********************************************

A DEM is usually represented with a map of its elevation, but this view can sometimes be challenging to interpret. 
Therefore, more visual representations were added into Demcompare to help analyse a DEM: Hillshade and sky-view factor representations. 
These additional visualizations are illustrated in the figure below.

The hillshade is a representation based on slope orientation. 
By convention, a north-oriented slope is assigned a value of 0°, an east-oriented slope 90°, a south-oriented slope 180°, and a west-oriented slope 270°. 
As a result, east-oriented slopes appear darker than west-oriented ones in this visualization.

This hillshade visualization not only helps users grasp the topography or 3D aspect of a DEM but also aids in identifying artifacts or other anomalous features.

On the other hand, the sky view factor assumes that the light is arriving from all the directions instead of coming from a punctual luminous source (the sun).
Then, a point is lit up proportionally to the solid angle of sky visible from its position. This makes it particularly easy to pinpoint features such as valleys. 
This representation serves as a valuable complement to the hillshade.

.. figure:: /images/HillshadeSVF.png
    :width: 1000px
    :align: center

.. _slope_normalized_elevation_difference:

Slope normalized elevation difference
*************************************

The most intuitive method used for comparing two DEMs is measuring the elevation difference pixel by pixel. 
However, this method has some biases due to the slope: the differences tend to be higher in the areas with steeper slopes. 
This phenomenon is illustrated in the plot below, generated using statistics from Demcompare for DEMs over a mountainous region. 
Such systemic error can be problematic. 
For instance, a DEM could have high errors in a flat zone that could be undetected because it would be dominated by the errors in a zone with important reliefs. 

.. figure:: /images/ErrorVSslope.png
    :width: 500px
    :align: center

To address the aforementioned bias, it seems relevant to normalize the elevation difference with the slope to remove this systemic error.
This normalization would facilitate a more accurate comparison of errors across pixels, regardless of their slope. 
[Reinartz]_ established a linear relation between the root mean square (rms) of the height difference and the tangent of the slope :math:`\alpha`:

.. math::

    rms = a + b \times \tan(\alpha)

To mitigate the influence of the slope, one can adjust the elevation error by dividing it by :math:`1 + b \times \tan(\alpha)`, where :math:`b` is calculated by computing a linear regression between the slope and the rms of the elevation difference. 
This will attenuate the bias and reveal the areas where the differences can actually be reduced as they would not result from the slope.

.. _angular_difference:

Angular difference
******************

While elevation differences—whether normalized by slope or not—can detect variations in height between two DEMs, they might not efficiently capture biases in their shapes. 
To address this limitation, a new method was implemented to compare the angle between the two normal vectors :math:`\vec{n_{k}}` of each pixel from two DEMs. 
This angle :math:`\theta` can be computed using the following relation:

.. math::

    \cos(\theta) = \frac{\vec{n_{1}} \cdot \vec{n_{2}}}{ \| \vec{n_{1}} \| \| \vec{n_{2}} \|}

A high angle indicates the presence of distortions and biases between the two DEMs.

.. _curvature:

Curvature
*********

Unlike metrics such as elevation difference or angular difference, curvature serves as a quality metric that doesn't rely on a reference DEM. 
This independence is crucial for several reasons:

* Not all reference DEMs boast a high spatial resolution.
* The reference DEM and the DEM under evaluation might represent different time frames, during which the area of interest could have experienced changes, be it from natural phenomena like landslides or human activities like construction.
* The two DEMs could differ in spatial resolution, and resampling a DEM can introduce errors and biases, especially concerning slopes and reliefs.

Given these potential discrepancies, it's important to incorporate standalone quality metrics like Curvature when evaluating a DEM. 

Curvature represents the second derivative of the elevation (the slope being the first derivative). 
For a given DEM :math:`u`, curvature is calculated with the formula :math:`div(\frac{\nabla u}{ \| \nabla u \|})`. 
Negative values are associated with convex profiles, whereas positive values are associated with concave profiles.

It is a really useful measure in urban areas as it highlights the quality of restitution of streets and buildings. 
For more natural areas, it can for instance bring some light on hydrologic networks or valleys. 

.. _slope_orientation_histogram:

Slope orientation histogram
***************************

Using the same convention with slopes as with the hillshade (0° for north orientation, 90° for east orientation…), it is possible to plot a circular distribution of the pixels according to their slope orientation. 
This quality measure is called the “slope orientation histogram” and it is very helpful in identifying the presence of artifacts.

For instance, it can detect if the method of DEM construction or sampling generates some slope orientations in the main direction of the interpolation grid. 
Therefore, it can spot the artefacts dependent on the DEM generation method, independently from the input data. 
An example is illustrated below (from Polidori et al, 2014) with 2 DEMs, one of them containing artifacts (SRTM) and the other being more realistic and homogeneous (Topodata).

.. figure:: /images/slopeOrientationHist.png
    :width: 500px
    :align: center

References
**********

For the linear relation between the root mean square (rms) of the height difference and the tangent of the slope :math:`\alpha`:

.. [Reinartz] Reinartz P, Pablo D, Krauss T, Poli D, Jacobsen K, Buyuksalih G. Benchmarking and quality analysis of DEM generated from high and very high resolution optical stereo satellite data. In Conference Proceedings: International Archives of Photogrammetry and Remote Sensing - ISSN: 1682-1777. Vol. XXXVIII. Enschede (The Netherlands): ITC; 2010. JRC57049
