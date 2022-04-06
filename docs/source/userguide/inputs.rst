.. _inputs:

Inputs
======


Configuration and parameters
****************************

Here is the list of the parameters of the input configuration file and its associated default value when it exists:



+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| Name                                                   | Description                                     | Type        | Default value       | Required |
+========================================================+=================================================+=============+=====================+==========+
| *outputDir*                                            | Output directory path                           | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM path*                                        | Path of the input DSM                           | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM zunit*                                       | Z axes unit of the input DSM                    | string      |       m             | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM georef*                                      | | Georef of the input DSM                       | string      |      "WGS84"        | No       |
|                                                        | | If set to "geoid", the according offset       |             |                     |          |
|                                                        | | will be added.                                |             |                     |          |
|                                                        | | If set to "geoid" and no "geoid_path"         |             |                     |          |
|                                                        | | is given, then EGM96 geoid                    |             |                     |          |
|                                                        | | will be used by default.                      |             |                     |          |
|                                                        | | Please note that this parameter is only       |             |                     |          |
|                                                        | | used if set to "geoid",                       |             |                     |          |
|                                                        | | since the .crs of the raster                  |             |                     |          |
|                                                        | | is used to obtain the georef.                 |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM geoid_path*                                  | Geoid path of the input DSM                     | string      |      None           | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM nodata*                                      | No data value of the input DSM                  | int         |        None         | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM roi*                                         | Processed Region of interest of the input DSM   | Dict        |        None         | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef path*                                        | Path of the input Ref                           | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef zunit*                                       | Z axes unit of the input Ref                    | string      |       m             | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef georef*                                      | | Georef of the input Ref                       | string      |      "WGS84"        | No       |
|                                                        | | If set to "geoid", the according offset       |             |                     |          |
|                                                        | | will be added.                                |             |                     |          |
|                                                        | | If set to "geoid" and no "geoid_path"         |             |                     |          |
|                                                        | | is given, then EGM96 geoid                    |             |                     |          |
|                                                        | | will be used by default.                      |             |                     |          |
|                                                        | | Please note that this parameter is only       |             |                     |          |
|                                                        | | used if set to "geoid",                       |             |                     |          |
|                                                        | | since the .crs of the raster                  |             |                     |          |
|                                                        | | is used to obtain the georef.                 |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef geoid_path*                                  | Geoid path of the input Ref                     | string      |    None             | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef nodata*                                      | No data value of the input Ref                  | int         |     None            | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *plani_opts corregistration_method*                    | Planimetric corregistration method              | string      | nuth_kaab           | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *plani_opts corregistration_iterations*                | Planimetric corregistration method              | int         | 6                   | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *plani_opts disp_init x*                               | | Planimetric corregistration                   | int         |  0                  | No       |
|                                                        | | initial disparity x                           |             |                     |          |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *plani_opts disp_init y*                               | | Planimetric corregistration                   | int         |  0                  | No       |
|                                                        | | initial disparity y                           |             |                     |          |
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



