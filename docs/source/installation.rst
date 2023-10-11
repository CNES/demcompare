.. highlight:: shell

============
Install
============


Quick pypi install
------------------


Create a Python virtual environment, git clone the repository and install.


.. code-block:: bash

  # Create your virtual environment "venv"
  python -m venv venv

  # Activate your venv (on UNIX)
  source venv/bin/activate

  # Update pip and setuptools package
  python -m pip --upgrade pip setuptools

  # Install the cars-mesh tool
  python -m pip install cars-mesh

  # Test if it works
  cars-mesh -h



Developer Install with Make
----------------------------

Git clone the repository, open a terminal and run the following commands:

.. code-block:: bash

    # Go to the desired folder
    cd /path/to/desired/folder

    # Clone repository
    git clone https://github.com/CNES/cars-mesh.git

    # Install
    make install

    # Activate your venv (on UNIX)
    # A flag "(NAME_OF_VENV)" should appear before your command line from now on
    source venv/bin/activate

    # Test if it works
    cars-mesh -h

It will install the virtual environment and all necessary to run the code.


Developer Install with local pip
---------------------------------

Create a Python virtual environment, git clone the repository and install the lib in dev mode (so to be able to modify
it dynamically).

.. code-block:: bash

    # Go to the desired folder where to save your virtual environment
    cd /path/to/desired/folder

    # Create your virtual environment and name it by replacing "NAME_OF_VENV" with whatever you like
    python -m venv NAME_OF_VENV

    # Activate your venv (on UNIX)
    # A flag "(NAME_OF_VENV)" should appear before your command line from now on
    source /path/to/desired/folder/NAME_OF_VENV/bin/activate

    # Update pip and setuptools package
    python -m pip install --upgrade pip setuptools

    # Clone library repository
    git clone https://github.com/CNES/cars-mesh.git

    # Install the cars-mesh lib in dev mode with the dev and doc tools
    python -m pip install -e .[dev,docs]

    # Test if it works
    cars-mesh -h
