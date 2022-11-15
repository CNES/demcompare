
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
    pip install demcompare

.. note::  In case of installation problems, please refer to :ref:`faq`

Command line execution
######################

Example of a basic DEM coregistration + statistics execution with the sample images and input configuration available on **demcompare** :

.. code-block:: bash

    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/images/srtm_sample.zip  # input stereo pair
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/json_conf_files/nuth_kaab_config.json # configuration file
    unzip srtm_sample.zip #uncompress data
    demcompare nuth_kaab_config.json #run demcompare

- For more information about **demcompare**'s command line execution, please refer to: :ref:`command_line_execution`
- For more information about **demcompare**'s steps, please refer to: :ref:`coregistration`, :ref:`statistics`, :ref:`report`