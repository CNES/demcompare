.. _command_line_execution:

.. role:: bash(code)
   :language: bash

Execution from the command line
===============================

**Demcompare** is executed from the command line with an input configuration file:

.. code-block:: bash

    demcompare config_file.json #run demcompare

The following code-block is an input configuration file example including 
both **coregistration**, **statistics** and **report** steps with demcompare data samples.

.. code-block:: json

    {
        "output_dir": "./test_output/",
        "input_ref": {
            "path": "./srtm_ref.tif"
        },
        "input_sec": {
            "path": "./srtm_blurred_and_shifted.tif"
        },
        "coregistration": {
            "method_name": "nuth_kaab_internal"
        },
        "statistics": {
            "alti-diff": {
                "remove_outliers": false
            }
        },
        "report" : "default"
    }

The **coregistration** and **report** steps are optional.
Remove one of them from the config file to prevent demcompare from running it.

The mandatory **statistics** step includes an **alti-diff** :ref:`DEM_processing_methods` section.
This **alti-diff** configuration allows to compute the difference in altitude between the two input DEMs.
All :ref:`DEM_processing_methods` can be found in :ref:`List of DEM processing methods <list_DEM_processing_methods>`.
They can be used one after the other in the **statistics** step.

The optional **report** step generates a report from computed statistics. 

Configuration parameters are described in associated sub-sections:

    - :ref:`input_dem`
    - :ref:`coregistration`
    - :ref:`statistics`
    - :ref:`report`
