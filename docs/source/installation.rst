.. highlight:: shell

============
Installation
============


Stable release
--------------

If deployed in Pypi, to install Mesh 3D, run this command in your terminal:

.. code-block:: console

    $ pip install mesh_3d

This is the preferred method to install Mesh 3D, as it will always install the most recent stable release.

Consider using a virtualenv to separate and test the installation.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Mesh 3D can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    # To update with real URL
    $ git clone git://github.com/CNES/mesh_3d

Or download the `tarball`_:

.. code-block:: console

    # To update with real URL
    $ curl -OJL https://github.com/CNES/mesh_3d/tarball/master

Once you have a copy of the source, you can install it in a virtualenv with:

.. code-block:: console

    $ make install
    $ source venv/bin/activate


.. _Github repo: https://github.com/CNES/mesh_3d
.. _tarball: https://github.com/CNES/mesh_3d/tarball/master
