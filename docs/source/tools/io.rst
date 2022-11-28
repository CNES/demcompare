.. _io:

===================
Input / Output (IO)
===================

A limited amount of input/output formats are handled:

* **Point cloud** : ply, las/laz
* **Mesh** : ply


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
