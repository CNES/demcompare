.. _simplify_mesh:

=============
Simplify mesh
=============

The meshing process creates a lot of simplexes (triangles in our case). All of them are not necessary to describe
the surface. A simplification step is needed to simplify further calculations.

Two approaches are implemented taken from open3d.

Quadric Error Metrics (Garland and Heckbert method)
===================================================

    | *Action* : "simplify_mesh"
    | *Method* : "garland-heckbert"

Garland and Heckbert method is based on an iterative vertex pairs contraction process while maintaining the overall
error rate.
This approach has the advantage of reconnecting regions that could have been separated (for example after a
BPA reconstruction).

Vertex Clustering
=================

    | *Action* : "simplify_mesh"
    | *Method* : "vertex_clustering"

The vertex clustering method pools all vertices that fall into a voxel of a given size to a single vertex.

.. warning::

    It does not seem to give any result (either good or bad). There might be an issue in the open3d code.
    Further studies should be conducted, or the method reimplemented or abandoned.
