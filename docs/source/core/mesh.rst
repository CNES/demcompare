.. _mesh:

====
Mesh
====

Meshing a point cloud consists in computing a surface from a set of points.

Three methods are implemented.


Ball Pivoting Algorithm (BPA)
=============================

    | *Action* : "mesh"
    | *Method* : "bpa"

The ball pivoting algorithm is a simple approach where a ball of a user defined radius is rolled over
the points of the cloud. Whenever it touches three points, a triangle is created.
Otherwise, a new seed point is chosen and the process starts over.
Thus, it creates surfaces with holes.


Poisson Reconstruction
======================

    | *Action* : "mesh"
    | *Method* : "poisson"

The surface is defined as the solution of a Poisson equation. The surface is reconstructed by
estimating the indicative function and by taking the isosurface.
As it is an optimisation process, it is robust to noise and creates a smooth surface. However,
it also moves points and tends to create too smooth surfaces.

.. warning::

    It changes the points of the cloud. Thus the pandas DataFrame and the open3d point cloud
    instance are no longer equal.

.. warning::
    The creation of outliers has been observed after using Poisson reconstruction.
    If so, texturing may not work. Indeed, the location of these outliers can be very
    bad and do not allow the completion of this step.


Delaunay Triangulation 2.5D
===========================

    | *Action* : "mesh"
    | *Method* : "delaunay_2d"

The concept of delaunay triangulation in 2D is to construct triangles so that no point of the cloud is ever
included in a circumscribed circle. It has some convenient characteristics including the fact of minimising the
number of sharp angles in triangles.

The 2.5D version consists in constructing a 2D delaunay triangulation ignoring the Z values, and afterwards adding it.

It thus has not the mathematical characteristics of a real delaunay triangulation. However, it is really fast, and
way faster than a real delaunay triangulation in 3D. It is also more consistent with our data that are in 2.5D.
