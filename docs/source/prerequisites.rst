.. highlight:: shell

=============
Prerequisites
=============

Input data is expected to follow some rules to be handled by the pipeline.


Point cloud/Mesh coordinate system
-----------------------------------

For the reconstruction pipeline, an input point or mesh is assumed to be expressed in an appropriate **UTM coordinate system**.
If it is not the case, one can use the function ``change_frame`` in ``tools/point_cloud_io.py`` to do the conversion.

For the evaluation pipeline, both inputs need to be in the same frame for consistency, but not necessarily in UTM (could be in Lambert93 for instance).
This frame should be chosen wisely since the euclidean distance needs to make sense in this frame for the metrics to be meaningful.


Input and output format
-----------------------

For point clouds, PLY, LAS and LAZ formats are handled. For meshes, only PLY format is supported.


(Optional) Pansharpened texture image
-------------------------------------

Basically, satellite product of type Pleiades is composed of two types of data:

* Panchromatic (gray scale, 50cm)
* Multispectral (R-G-B-Nir, 2m)

To have a high resolution colored texture, one needs to apply pansharpening to fuse these data.
A way to do it is to use the ``BundleToPerfectSensor`` function from the `Orfeo Toolbox <https://www.orfeo-toolbox.org/packages/doc/tests-rfc-52/cookbook-3b41671/Applications/app_BundleToPerfectSensor.html>`_ and use the RPC associated with the panchromatic image.


