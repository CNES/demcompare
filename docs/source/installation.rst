.. highlight:: shell

============
Install
============


Quick installation via CMake
-----------------------------

Git clone the repository, open a terminal and launch the following commands:

.. code-block:: bash

    # Go to the desired folder
    cd /path/to/desired/folder

    # Clone repository
    # Make sure to check the right way to do it, whether you are internal or external to CNES
    # Internal: https://confluence.cnes.fr/pages/viewpage.action?pageId=26166114
    # External: https://confluence.cnes.fr/pages/viewpage.action?pageId=26159013
    git clone git@gitlab.cnes.fr:cars/etudes/rt_mesh_3d.git .

    # Install
    make install

    # Activate your venv (on UNIX)
    # A flag "(NAME_OF_VENV)" should appear before your command line from now on
    source /path/to/desired/folder/NAME_OF_VENV/bin/activate

    # Test if it works
    mesh_3d -h

It will install the virtual environment and all necessary to run the code.


Quick manual installation
-------------------------

Create a Python virtual environment, git clone the repository and install the lib in dev mode (so to be able to modify
it dynamically).

.. code-block:: bash

    # Go to the desired folder where to save your virtual environment
    cd /path/to/desired/folder

    # Create your virtual environment and name it by replacing "NAME_OF_VIRTUALENV" with whatever you like
    python -m venv NAME_OF_VENV

    # Activate your venv (on UNIX)
    # A flag "(NAME_OF_VENV)" should appear before your command line from now on
    source /path/to/desired/folder/NAME_OF_VENV/bin/activate

    # Update pip and setuptools package
    python -m pip --upgrade pip setuptools

    # Clone library repository
    # Make sure to check the right way to do it, whether you are internal or external to CNES
    # Internal: https://confluence.cnes.fr/pages/viewpage.action?pageId=26166114
    # External: https://confluence.cnes.fr/pages/viewpage.action?pageId=26159013
    git clone git@gitlab.cnes.fr:cars/etudes/rt_mesh_3d.git .

    # Install the mesh_3d lib in dev mode with the dev and doc tools
    python -m pip install -e .[dev,docs]

    # Test if it works
    mesh_3d -h
