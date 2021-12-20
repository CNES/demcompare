Install
=======


Only Linux Plaforms are supported (virtualenv or bare machine) with Python 3 installed.
The default install mode is via the pip package, typically through a virtualenv:
.. code-block:: bash
    python3 -m venv venv
    source venv/bin/activate
    pip install demcompare

Developer mode
This package can be installed through the following commands:

.. code-block:: bash
    git clone https://github.com/CNES/demcompare
    cd demcompare
    make install
    source venv/bin/activate # to go in installed dev environment

Dependencies : git, make

Troubleshootings
Depending on pip version, installation problems can happen with packages dependencies installation order. Install and upgrade pip, numpy and cython if demcompare installation crashes:

.. code-block:: bash
    python3 -m pip install --upgrade pip
    python3 -m pip install --upgrade numpy cython

Be careful: Rasterio have its own embedded version of GDAL
Please use rasterio no-binary version in Makefile install if you want to use a GDAL local version:

.. code-block:: bash
    python3 -m pip install --no-binary rasterio rasterio



Dependencies
************
The full list of dependencies can be observed from the [setup.cfg](./setup.cfg) file.
