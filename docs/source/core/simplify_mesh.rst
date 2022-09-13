.. _simplify_mesh:

=============
Simplify mesh
=============

The meshing process creates a lot of simplexes (triangles in our case). All of them are not necessary to describe
the surface. A simplification step is needed to simplify further calculations.

Two approaches are implemetend taken from open3d.

Quadric Error Metrics (Garland and Heckbert method)
===================================================

Garland and Heckbert method is based on an iterative vertex pairs contraction process while maintaining the overall
error rate.
This approach has the advantage of reconnecting regions that could have been seperated (for example after a
BPA reconstruction).

Vertex Clustering
=================

The vertex clustering method pools all vertices that fall into a voxel of a given size to a single vertex.

.. warning::

    It does not seem to give any result (either good or bad). Further study should be conducted, or the method
    abandoned.
