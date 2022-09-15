.. _filter:

==================
Filter point cloud
==================

Point cloud filtering consists in removing outliers from the point cloud.
Outliers are defined as points that are significantly higher or lower from the average surface.
There are generated during the correlation process on regions with matching problems (shadows, homogenous textures, etc.).
Outliers ought to be removed from the cloud because they bring false information. However, in satellite images,
outliers are often gathered in clusters. Thus, since photogrammetric point clouds are not dense, removing outliers leaves holes in the data.
How to deal with these clusters is an open question (leave it that way? re-densify the data by interpolation? etc.)

Four approaches are proposed.


Statistical filtering
=====================

    | *Action* : "filter"
    | *Method* : "statistics_o3d"

Statistical filtering consists in computing statistics over the point cloud, i.e. the average distance to the k nearest neighbours and the standard deviation of this distance.
Then, for each point of the cloud, its statistics are compared to the overall ones.
If they deflect over a user defined threshold (specified as a factor over the std), the point is classified as outlier.

.. note::

    It can easily be parallelized since each test on point is indepedent from the others.


Local density analysis
======================

    | *Action* : "filter"
    | *Method* : "local_density_analysis"

Based on the work of Ning et al., this method aims at computing the probabiity of a point to be an outlier
considering its local density for a defined number of neighbours.  If its local density is too low, this point is
considered as an outlier.

Source: Ning, X., Li, F., Tian, G., & Wang, Y. (2018). An efficient outlier removal method for scattered point cloud data.
PLoS ONE, 13.

.. note::

    It can easily be parallelized since each test on point is indepedent from the others.


Radius filtering
================

    | *Action* : "filter"
    | *Method* : "radius_o3d"

Radius filtering is the dual method not relying on nearest neighbours but on a ball of specified radius.
The user defined the number of neighbours that a point should have in a sphere of user-defined radius.
Then, each point of the cloud is tested, and if it does not satisfy this requirement, it is classified as outlier.

.. note::

    It can easily be parallelized since each test on point is indepedent from the others.


Radius filtering by cluster
===========================

    | *Not implemented anymore*

This approach is implemented in CARS. Briefly, it is the same idea as the radius filtering method.
However, points are first gathered in clusters that are flagged as outliers if they do not have enough points.

.. note::

    This method can hardly be parallelized because lists of points are updated during the neighbours' search making
    the approach dynamic.

.. warning::

    It is now deprecated in the code since CARS v4. It is given here as an information notice.
