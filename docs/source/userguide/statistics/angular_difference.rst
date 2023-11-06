.. _angular_difference:

Angular difference
==================

While elevation differences—whether normalized by slope or not—can detect variations in height between two DEMs, they might not efficiently capture biases in their shapes. 
To address this limitation, a new method was implemented to compare the angle between the two normal vectors :math:`\vec{n_{k}}` of each pixel from two DEMs. 
This angle :math:`\theta` can be computed using the following relation:

.. math::

    \cos(\theta) = \frac{\vec{n_{1}} \cdot \vec{n_{2}}}{ \| \vec{n_{1}} \| \| \vec{n_{2}} \|}

A high angle indicates the presence of distortions and biases between the two DEMs.
This can highlight some discrepancies not distinguishable with the simple elevation difference. 
For instance, in the figure bellow, high differences can be observed in some areas. 
In particular the straight line in the top left-hand corner is not visible with the elevation difference and corresponds in fact to a power line or a ski lift which is visible in the reference DEM but not in the second DEM generated from satellite optical images.

.. figure:: /images/AngularDiff.png
    :width: 500px
    :align: center

    Angular difference over a DEM covering a mountainous region in the Alps. White noisy areas correspond to no-data pixels.