
.. role:: bash(code)
   :language: bash

Getting started
===============

Install
#######

Demcompare is available on Pypi and can be installed by:

.. code-block:: bash

    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install demcompare

.. note::  In case of installation problems, please refer to :ref:`faq`

Command line execution
######################

Example of a basic DEM coregistration + statistics execution with the sample images and input configuration available on demcompare :

.. code-block:: bash

   # download data samples
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/srtm_blurred_and_shifted.tif
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/srtm_ref.tif

    # download demcompare predefined configuration
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/sample_config.json

    # run demcompare
    demcompare sample_config.json

Another configuration is also available, allowing to compute more interesting metrics:

.. code-block:: bash

    # download the other demcompare predefined configuration file, 
    # this one allows to compute all the interesting metrics available for comparing the 2 input DEMs
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/sample_config_full.json
    # run demcompare
    demcompare sample_config_full.json

For more details, please refer to :ref:`command_line_execution` section.