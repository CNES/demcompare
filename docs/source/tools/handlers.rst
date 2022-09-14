.. _handlers:

========
Handlers
========

Handlers are tooling functions to manipulate point clouds and meshes easily. A class for each object is created and
provides a bunch of useful functions, as well as the serialization and deserialization functions.


PointCloud class
================

Point cloud information is handled in a pandas DataFrame (point coordinates, colours, normal coordinates, classes,
etc.).

Part of the information is at some point initialized in an open3D point cloud object when processings require it.
However, these objects are limited (8 bits RGB colours, etc.)

.. warning::

    No consistency check is done between the pandas DataFrame and open3D point cloud object.
    It means that at any time one can be updated but not the other.


Mesh class
==========

A Mesh object is composed of a PointCloud attribute and a list of triangles represented by the indexes of the vertices
in the point cloud. This list is captured in a pandas DataFrame. In the same way as the PointCloud class, an
open3d mesh object can be initialized for some processings.

The same remarks and warnings apply.

