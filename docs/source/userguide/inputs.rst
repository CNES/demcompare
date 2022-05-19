.. _inputs:

Inputs
======


Configuration and parameters
****************************

Here is the list of the parameters of the input configuration file and its associated default value when it exists:



+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| Name                                                   | Description                                     | Type        | Default value       | Required |
+========================================================+=================================================+=============+=====================+==========+
| *output_dir*                                           | Output directory path                           | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_dem_to_align path*                              | Path of the input DSM                           | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_dem_to_align zunit*                             | Z axes unit of the input DSM                    | string      |       m             | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_dem_to_align geoid_georef*                      | | True if the georef of the input DSM           | bool        |     False           | No       |
|                                                        | | is "geoid". In that case, the according offset|             |                     |          |
|                                                        | | will be added to the .crs of the raster.      |             |                     |          |
|                                                        | | If set to "geoid_georef" and no "geoid_path"  |             |                     |          |
|                                                        | | is given, then EGM96 geoid                    |             |                     |          |
|                                                        | | will be used by default.                      |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_dem_to_align geoid_path*                        | Geoid path of the input DSM                     | string      |      None           | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_dem_to_align nodata*                            | No data value of the input DSM                  | int         |        None         | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_dem_to_align roi*                               | Processed Region of interest of the input DSM   | Dict        |        None         | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_ref path*                                       | Path of the input Ref                           | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_ref zunit*                                      | Z axes unit of the input Ref                    | string      |       m             | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_ref geoid_georef*                               | | True if the georef of the input Ref           | bool        |     False           | No       |
|                                                        | | is "geoid". In that case, the according offset|             |                     |          |
|                                                        | | will be added to the .crs of the raster.      |             |                     |          |
|                                                        | | If set to "geoid_georef" and no "geoid_path"  |             |                     |          |
|                                                        | | is given, then EGM96 geoid                    |             |                     |          |
|                                                        | | will be used by default.                      |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_ref geoid_path*                                 | Geoid path of the input Ref                     | string      |    None             | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *input_ref nodata*                                     | No data value of the input Ref                  | int         |     None            | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *coregistration method_name*                           | Planimetric coregistration method               | string      | nuth_kaab           | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *coregistration number_of_iterations*                  | | Number of iterations                          | int         | 6                   | No       |
|                                                        | | of the coregistration method                  |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *coregistration estimated_initial_shift_x*             | | Estimated initial x                           | int         |  0                  | No       |
|                                                        | | coregistration shift                          |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *coregistration estimated_initial_shift_y*             | | Estimated initial y                           | int         |  0                  | No       |
|                                                        | | coregistration shift                          |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts elevation_thresholds list*                 | | List of elevation thresholds for              | list[float] |[0.5, 1, 3]          | No       |
|                                                        | | statistics                                    |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| | *stats_opts elevation_thresholds*                    | zunit of the elevation thresholds               | string      | m                   | No       |
| | *zunit*                                              |                                                 |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| | *stats_opts*                                         | | Slope ranges for classification               | list[int]   | [0, 10, 25, 50, 90] | No       |
| | *to_be_classification_layers*                        | | layers                                        |             |                     |          |
| | *slope ranges*                                       |                                                 |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| | *stats_opts*                                         | | Slope reference for classification            | string      | None                | No       |
| | *to_be_classification_layers*                        | | layers                                        |             |                     |          |
| | *slope ref*                                          |                                                 |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| | *stats_opts*                                         | Slope dsm for classification layers             | string      | None                | No       |
| | *to_be_classification_layers*                        |                                                 |             |                     |          |
| | *slope dsm*                                          |                                                 |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts remove_outliers*                           | | Remove outliers during statistics             | bool        | False               | No       |
|                                                        | | computation                                   |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts plot_real_hists*                           | Plot histograms                                 | bool        | True                | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts alti_error_threshold value*                | Altimetric error threshold value                | float       | 0.1                 | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts alti_error_threshold unit*                 | Altimetric error threshold unit                 | string      | m                   | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| | *stats_opts*                                         | | Classification layer called                   | string      |                     | No       |
| | *classification_layers*                              | | *name* 's ref                                 |             |                     |          |
| | *name* *ref*                                         |                                                 |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| | *stats_opts*                                         | | Classification layer called                   | string      |                     | No       |
| | *classification_layers*                              | | *name* 's dsm                                 |             |                     |          |
| | *name* *dsm*                                         |                                                 |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| | *stats_opts*                                         | | Classification layer called                   | Dict        |                     | No       |
| | *classification_layers*                              | | *name* 's classes                             |             |                     |          |
| | *name* *classes*                                     |                                                 |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+

Input format and examples
*************************
.. _inputs_reference:

**Demcompare** requires the input images to be a geotiff file **.tif** which contains the :term:`DSM` in the required cartographic projection.

The DEMs altitudes can rely on both ellipsoid and geoid references. For instance, if DEMs altitudes are to rely on **geoid**, configurations could be:

.. sourcecode:: text

    "inputDSM" : {  "path": "./inputDSM.tif"
                    "zunit" : "meter",
                    "georef" : "geoid",
                    "nodata" : }

In this case, **EGM96 geoid** will be used by default.

Otherwise, the absolute path to a locally available geoid model can be given. The geoid local model should be either a *GTX*, *NRCAN* or *NTv2* file.

For instance, if DEMs altitudes are to rely on a local *.gtx* available **geoid** model, configurations could be:

.. sourcecode:: text

    "inputDSM" : {  "path": "./inputDSM.tif"
                    "zunit" : "meter",
                    "georef" : "geoid",
                    "geoid_path": "path/to/egm08_25.gtx"
                    "nodata" : }



