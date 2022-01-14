
.. role:: bash(code)
   :language: bash

Getting started
===============

Install
#######

For information about demcompare's installation please refer to: :ref:`install`

Quick start
###########

.. code-block:: bash

    pip install demcompare #install demcompare latest release
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/images/strm_sample.zip  # input stereo pair
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/json_conf_files/nuth_kaab_config.json # configuration file
    unzip strm_sample.zip #uncompress data
    demcompare nuth_kaab_config.json #run demcompare


The results can be observed with:

.. code-block:: bash

    firefox test_output/doc/published_report/html/demcompare_report.html &



Advanced Quick start
####################

demcompare allows one to execute only a subset of the whole process. As such, as :bash:`--step` command line argument is
provided. It accepts values in `coregistration` `stats` `report` :

.. code-block:: bash

    pip install demcompare #install demcompare latest release
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/images/strm_sample.zip  # input stereo pair
    wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/json_conf_files/nuth_kaab_config.json # configuration file
    unzip strm_sample.zip #uncompress data
    demcompare --step coregistration nuth_kaab_config.json #run demcompare coregistration step
    demcompare --step stats nuth_kaab_config.json #run demcompare stats step

- All the steps but **stats** are optional. Coregistration step is not mandatory for stats and following steps as one can decide its DEMs are already coregistered.

- demcompare can start at any step as long as previously required steps have been launched.

  - This means that one can launch the report step only as long as the stats step has already been performed from a previous demcompare launch and the *config.json* remains the same. The steps are space-separated (no comma).

- For more more information about demcompare's steps, pleare refer to: :ref:`step_by_step`