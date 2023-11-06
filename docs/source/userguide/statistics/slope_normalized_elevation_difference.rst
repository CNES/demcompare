.. _slope_normalized_elevation_difference:

Slope normalized elevation difference
=====================================

The most intuitive method used for comparing two DEMs is measuring the elevation difference pixel by pixel. 
However, this method has some biases due to the slope: the differences tend to be higher in the areas with steeper slopes. 
This phenomenon is illustrated in the plot below, generated using statistics from demcompare for DEMs over a mountainous region. 
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

References
**********

For the linear relation between the root mean square (rms) of the height difference and the tangent of the slope :math:`\alpha`:

.. [Reinartz] Reinartz P, Pablo D, Krauss T, Poli D, Jacobsen K, Buyuksalih G. Benchmarking and quality analysis of DEM generated from high and very high resolution optical stereo satellite data. In Conference Proceedings: International Archives of Photogrammetry and Remote Sensing - ISSN: 1682-1777. Vol. XXXVIII. Enschede (The Netherlands): ITC; 2010. JRC57049