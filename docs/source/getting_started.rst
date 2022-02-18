
.. role:: bash(code)
   :language: bash

Getting started
===============

Install
#######

Demcompare is available on Pypi and can be installed by:

.. code-block:: bash

    pip install demcompare

For more information about **demcompare**'s installation please refer to: :ref:`install`

Quick start
###########


Example of a basic DSM comparison execution with the sample images and input configuration available on **demcompare** :

.. code-block:: bash

    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/images/srtm_sample.zip  # input stereo pair
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/json_conf_files/nuth_kaab_config.json # configuration file
    unzip srtm_sample.zip #uncompress data
    demcompare nuth_kaab_config.json #run demcompare


The results can be observed with:

.. code-block:: bash

    firefox test_output/doc/published_report/html/demcompare_report.html &



Advanced Quick start
####################

**Demcompare** allows one to execute only a subset of the whole process using the command line. As such, a :bash:`--step` command line argument is
provided. It accepts the steps: *coregistration*, *stats* and *report* :

.. code-block:: bash

    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/images/strm_sample.zip  # input stereo pair
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/json_conf_files/nuth_kaab_config.json # configuration file
    unzip strm_sample.zip #uncompress data
    demcompare --step coregistration nuth_kaab_config.json #run demcompare coregistration step
    demcompare --step stats nuth_kaab_config.json #run demcompare stats step

- For more information about **demcompare**'s command line execution, please refer to: :ref:`command_line_execution`
- For more information about **demcompare**'s steps, please refer to: :ref:`step_by_step`