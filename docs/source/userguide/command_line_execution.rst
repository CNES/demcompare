.. _command_line_execution:

.. role:: bash(code)
   :language: bash

Execution from the command line
===============================

**Demcompare** is executed from the command line with an input configuration file:

.. code-block:: bash

    demcompare config_file.json #run demcompare

The following code-block is an input configuration file example including 
both **coregistration** and **statistics** steps. These steps are optional.
Remove one of them from the config file to prevent demcompare from running it.

An optional **report** step is included to generate a report if statistics are computed. 

.. code-block:: json

    "output_dir": "./test_output/",
    "input_ref": {
        "path": "./Gironde.tif",
    },
    "input_sec": {
        "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
      },
    "coregistration": {
        "coregistration_method": "nuth_kaab_internal",
    }
    "statistics": {
        "remove_outliers": "False",
    }
    "report" : "default"

Configuration parameters are described in associated sub-sections:

    - :ref:`input_dem`
    - :ref:`coregistration`
    - :ref:`statistics`
    - :ref:`report`
