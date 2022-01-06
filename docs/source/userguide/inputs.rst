.. _inputs:

Inputs
======

Configuration parameters
************************



Configuration and parameters
****************************

Here is the list of the parameters and the associated default value when it exists:



+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| Name                                                   | Description                                     | Type        | Default value       | Required |
+==========================================================================================================+=============+=====================+==========+
| *outputDir*                                            | Output directory path                           | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM path*                                        | Path of the input DSM                           | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM zunit*                                       | Z axes unit of the input DSM                    | string      |       m             | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM georef*                                      | Georef of the input DSM                         | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM geoid_path*                                  | Geoid path of the input DSM                     | string      |      None           | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputDSM nodata*                                      | No data value of the input DSM                  | int         |        None         | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef*   path*                                     | Path of the input Ref                           | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef zunit*                                       | Z axes unit of the input Ref                    | string      |       m             | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef georef*                                      | Georef of the input Ref                         | string      |                     | Yes      |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef geoid_path*                                  | Geoid path of the input Ref                     | string      |    None             | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *inputRef nodata*                                      | No data value of the input Ref                  | int         |     None            | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *plani_opts corregistration_method*                    | Planimetric corregistration method              | string      | nuth_kaab           | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *plani_opts corregistration_iterations*                | Planimetric corregistration method              | int         | 6                   | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *plani_opts disp_init x*                               | Planimetric corregistration initial disparity x | int         |  0                  | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *plani_opts disp_init y*                               | Planimetric corregistration initial disparity y | int         |  0                  | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts elevation_thresholds list*                 | List of elevation thresholds for statistics     | list[float] |[0.5, 1, 3]          | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts elevation_thresholds zunit*                | zunit of the elevation thresholds               | string      | m                   | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts to_be_classification_layers slope ranges*  | Slope ranges for classification layers          | list[int]   | [0, 10, 25, 50, 90] | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts to_be_classification_layers slope ref*     | Slope reference for classification layers       | string      | None                | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts to_be_classification_layers slope dsm*     | Slope dsm for classification layers             | string      | None                | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts remove_outliers*                           | Remove outliers during statistics computation   | bool        | False               | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts plot_real_hists*                           | Plot histograms                                 | bool        | True                | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts alti_error_threshold value*                | Altimetric error threshold value                | float       | 0.1                 | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts alti_error_threshold unit*                 | Altimetric error threshold unit                 | string      | m                   | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts classification_layers *name* ref*          | Classification layer called *name* 's ref       | string      |                     | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts classification_layers *name* dsm*          | Classification layer called *name* 's dsm       | string      |                     | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+
| *stats_opts classification_layers *name* classes*      | Classification layer called *name* 's classes   | Dict        |                     | No       |
+--------------------------------------------------------+-------------------------------------------------+-------------+---------------------+----------+


If DEMs altitudes are to rely on **geoid**, configurations could be:

.. sourcecode:: text

    "inputDSM" : {  "path", "./inputDSM.tif"
                            "zunit" : "meter",
                            "georef" : "geoid",
                            "nodata" : }

In this case, **EGM96 geoid** will be used by default.

Otherwise, the absolute path to a locally available geoid model can be given, for instance:

.. sourcecode:: text

    "inputDSM" : {  "path", "./inputDSM.tif"
                            "zunit" : "meter",
                            "georef" : "geoid",
                            "geoid_path": "path/to/egm08_25.gtx"
                            "nodata" : }


