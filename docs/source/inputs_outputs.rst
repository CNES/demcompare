.. highlight:: shell

====================
Inputs/Outputs 
====================

Input and output format
-----------------------

For point clouds, PLY, LAS and LAZ formats are handled. For meshes, only PLY format is supported.

.. note::

    Visualisation should be done with CloudCompare, MeshLab or potree. Other applications may not work.
    If the coloured point cloud appears black (or mono-coloured), it might be because the colours
    were not normalized as 8bits. RGB bands should be processed before visualization.


.. warning::

    If some texture is saved, the texture image will be saved in the same directory as the mesh file. It should remain
    here for visualisation softwares to be able to read it at least for PLY format.


.. warning::

    OBJ format for mesh or textured mesh with open3d has a bug. It is reported and shall be fixed soon.
    However, for now, if one wants to save a mesh in a OBJ format, one should implement its serializer/deserializer.
    (It could be interesting to have a look at the OpenDroneMap project to check how they handle it.)


Point cloud/Mesh coordinate system
-----------------------------------

Moreover, input data is expected to follow some rules to be handled by the pipeline.

For the reconstruction pipeline, an input point or mesh is assumed to be expressed in an appropriate **UTM coordinate system**.
If it is not the case, one can use the function ``change_frame`` in ``tools/point_cloud_io.py`` to do the conversion.

For the evaluation pipeline, both inputs need to be in the same frame for consistency, but not necessarily in UTM (could be in Lambert93 for instance).
This frame should be chosen wisely since the euclidean distance needs to make sense in this frame for the metrics to be meaningful.


(Optional) Pansharpened texture image
-------------------------------------

Basically, satellite product of type Pleiades is composed of two types of data:

* Panchromatic (gray scale, 50cm)
* Multispectral (R-G-B-Nir, 2m)

To have a high resolution colored texture, one needs to apply pansharpening to fuse these data.
A way to do it is to use the ``BundleToPerfectSensor`` function from the `Orfeo Toolbox <https://www.orfeo-toolbox.org/packages/doc/tests-rfc-52/cookbook-3b41671/Applications/app_BundleToPerfectSensor.html>`_ and use the RPC associated with the panchromatic image.


