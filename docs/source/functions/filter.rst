.. _filter:

==================
Filter point cloud
==================

Point cloud filtering consists in removing outliers from the point cloud.
Outliers are defined as points that are significantly higher or lower from the average surface.
There are generated during the correlation process on regions with matching problems (shadows,
homogeneous textures, etc.).
Outliers ought to be removed from the cloud because they propagate false information. However, in
satellite images,
outliers are often gathered in clusters. Thus, since photogrammetric point clouds are not dense,
removing outliers leaves holes in the data.
How to deal with these clusters is an open question (leave it that way? re-densify the data by
interpolation? etc.)

Four approaches are proposed.


Statistical filtering
=====================

    | *Action* : "filter"
    | *Method* : "statistics_o3d"

Statistical filtering consists in computing statistics over the point cloud, i.e. the average
distance to the k nearest neighbours :math:`\bar{d}` and the mean standard deviation of this
distance :math:`\bar{\sigma}`.
Then, for each point of the cloud, its statistics are compared to the overall ones.
If they deflect over a user defined threshold (specified as a factor over the std), the point
is classified as outlier.

Let :math:`P` be a point of the cloud and :math:`\bar{d}_P` the mean distance to its k nearest neighbours.
Let :math:`s` be a user defined parameter. Then:

.. math::

    P \ \text{outlier} \ \Longleftrightarrow \ \bar{d}_P > \bar{d} + s \cdot \bar{\sigma}

.. note::

    It can easily be parallelized since each test on point is independent from the others.


Local density analysis
======================

    | *Action* : "filter"
    | *Method* : "local_density_analysis"

Based on the work of Ning et al., this method aims at computing the probability of a point to be
an outlier considering its local density for a defined number of neighbours.  If its local
density is too low, this point is considered as an outlier.

Let :math:`LD` be the local density function for a point :math:`P` considering
its :math:`k` nearest neighbours (knn). It is defined by:

.. math::

    LD(P) = \frac{1}{k} \sum_{Q_j \in knn(P)} \exp \left( \frac{-d(P, Q_j)}{\bar{d}(P)} \right) \\
    \text{with} \ \bar{d}(P) = \frac{1}{k} \sum_{j=1}^k d(P, Q_j) \\ \text{and} \ d(P,Q_j) = ||P-Q_j||_2

Let :math:`prob(P) = 1 - LD(P)` be the probability of a point :math:`P` to be an outlier. Furthermore,
let :math:`\delta` be a user defined threshold. Thus:

.. math::

    P \ \text{outlier} \ \Longleftrightarrow \ prob(P) > \delta

.. note::

    It can easily be parallelized since each test on point is independent from the others.

Source: Ning, X., Li, F., Tian, G., & Wang, Y. (2018). An efficient outlier removal method for
scattered point cloud data. PLoS ONE, 13.

Radius filtering
================

    | *Action* : "filter"
    | *Method* : "radius_o3d"

Radius filtering is the dual method to statistical filtering not relying on nearest neighbours but on a ball of specified radius.
The user defines the number of neighbours that a point should have in a sphere of user-defined radius.
Then, each point of the cloud is tested, and if it does not satisfy this requirement, it is classified as outlier.

Let :math:`N_{min}` be the minimal number of neighbours in a ball of radius :math:`R` defined by the
user. Let :math:`N` be the number of neighbours of point :math:`P` defined in a ball of radius
:math:`R`. Then:

.. math::

    P \ \text{outlier} \ \Longleftrightarrow \ N < N_{min}

.. note::

    It can easily be parallelized since each test on point is independent from the others.

.. note::

    A version is available in CARS code (but not used here.



