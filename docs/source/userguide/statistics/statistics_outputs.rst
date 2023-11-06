.. _statistics_outputs:

Statistics outputs
==================

Output files and their required parameters
******************************************

The images and files saved with the ``statistics`` option activated on the configuration :


+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| Name                                                               | Description                                                                              |
+====================================================================+==========================================================================================+
| *dem_for_stats.tif*                                                | DEM on which the statistics have been computed                                           |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| *ref and sec_rectified_support_map.tif*                            | | Stored on each classification layer folder, the rectified support maps                 |
|                                                                    | | where each pixel has a class value.                                                    |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| *stats_results.csv and .json*                                      | | Stored on each classification layer folder,                                            |
|                                                                    | | the CSV and Json files storing the computed statistics by class.                       |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| *stats_results_intersection.csv and .json*                         | | Stored on each classification layer folder, the CSV and Json files                     |
|                                                                    | | storing the computed statistics by class in mode intersection.                         |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| *stats_results_exclusion.csv and .json*                            | | Stored on each classification layer folder, the CSV and Json files                     |
|                                                                    | | storing the computed statistics by class in mode exclusion.                            |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+

Output directories
******************

With the command line execution, the following statistics directories that may store the respective files will be automatically generated.

One output directory per **DEM processing method** is created:

.. code-block:: bash

    .output_dir
    +-- stats
        +-- *dem_processing_method*
            +-- dem_for_stats.tif
            +-- *classification_layer_name*
                +-- stats_results.json/csv
                +-- stats_results_intersection.json/csv
                +-- stats_results_exclusion.json/csv
                +-- ref_rectified_support_map.tif
                +-- sec_rectified_support_map.tif

.. note::
    Please notice that even if no classification layer has been specified, the results will be stored in a folder called ``global``, as it
    is the classification layer that is always computed and only considers all valid pixels.

.. note::
    Please notice that some data may be missing if it has not been computed for the classification layer (ie. intersection maps are only computed under certain conditions :ref:`modes`).
