.. _mesh:

====
Mesh
====

Mesh a point cloud consist in computing a surface from a set of points.

Three methods are implemented.


Ball Pivoting Algorithm (BPA)
=============================

The ball pivoting algorithm is a simple approach where a ball of a user defined radius is rolled over
the points of the cloud. Whenever it touches three points, a triangle is  created.
Otherwise, a new seed point is chosen and the process starts over.
Thus, it creates surfaces with holes.


Poisson Reconstruction
======================

The surface is defined as the solution of a Poisson equation. The surface is reconstructed by estimating the indicative
function and by taking the isosurface.
As it is an optimisation process, it is robust to noise and creates a smooth surface. However, it also moves points
and tends to create too smooth surfaces.


Delaunay Triangulation 2.5D
===========================

The concept of delaunay triangulation in 2D is to construct triangles so that no point of the cloud is ever
included in a circumscribed circle. It has some pretty characteristics including the fact of minimising the
number sharp angles in triangles.

The 2.5D version consists in constructing a delaunay triangulation 2D ignoring the Z values, and afterwards adding it.

It thus has not the mathematical characteristics of a real delaunay triangulation. However, it is really fast, and
way faster than a real delaunay triangulation in 3D. It is also more consistent with our data that are in 2.5D.
