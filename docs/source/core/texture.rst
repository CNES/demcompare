.. _texture:

=======
Texture
=======

    | *Action* : "texture"
    | *Method* : "texturing"


Texturing consists in associating each simplex of the surface to a part of an image.

To apply a georeferenced texture to our mesh, we use inverse RPC to get the image coordinates of each point of
the cloud.

.. warning::

    If OpenGL is used in the back to display the texture, one needs to normalize the image coordinates between 0 and 1.
    Otherwise, over 1, it is interpreted as a duplication of the texture. The code makes this normalization.

In the current version of the code, occlusions are not handled; neither are multi-image textures.
Some limitations come from the file format on which to serialize the mesh which does not handle it.
File format and texture should closely be considered.

.. note::

    Currently, the code uses an independent implementation of RPC. It could be replaced by any other application
    (ShareLoc, LibGeo, etc.).

.. warning::

    The current implementation of RPC is Pleiades oriented. One should check the `tools/rpc` code and modify it for
    other data.
