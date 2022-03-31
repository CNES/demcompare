.. _outputs:

Outputs
=======


**Demcompare** will store several data on the *"OutputDir"* folder which is specified on the input configuration file. Here is a brief explanation for each one.

The images :

+----------------------------------+------------------------------------------------------------------------------------------+
| Name                             | Description                                                                              |
+==================================+==========================================================================================+
| *intial_dh.tif*                  | | Altitude differences image when both DEMs have been reprojected                        |
|                                  | | to the same inputDSM grid and no coregistration has been performed                     |
+----------------------------------+------------------------------------------------------------------------------------------+
| *final_dh.tif*                   | | Altitude differences image from the reprojected DEMs after                             |
|                                  | | the coregistration.                                                                    |
+----------------------------------+------------------------------------------------------------------------------------------+
| *dh_col_wise_wave_detection.tif* | | Image computed by substituting the `final_dh.tif` average col                          |
|                                  | | to final_dh.tif itself. It helps to detect any residual oscillation.                   |
+----------------------------------+------------------------------------------------------------------------------------------+
| *dh_row_wise_wave_detection.tif* | | Image computed by substituting the `final_dh.tif` average row                          |
|                                  | | to final_dh.tif itself. It helps to detect any residual oscillation.                   |
+----------------------------------+------------------------------------------------------------------------------------------+
| *coreg_DSM.tif*                  | | Intermediate coregistered DSM used for the stats computation. Both coregistered        |
|                                  | | DEMs will have the dsm’s georeference grid and an intermediate                         |
|                                  | | georeference origin.                                                                   |
+----------------------------------+------------------------------------------------------------------------------------------+
| *coreg_Ref.tif*                  | | Intermediate coregistered Ref used for the stats computation. Both coregistered        |
|                                  | | DEMs will have the dsm’s georeference grid and an intermediate                         |
|                                  | | georeference origin.                                                                   |
+----------------------------------+------------------------------------------------------------------------------------------+
| *AltiErrors.tif*                 | | The images whose names start with **AltiErrors-** are the plots saved by               |
|                                  | | demcompare. They show histograms by stats set and the same                             |
|                                  | | histogramsfitted by gaussian.                                                          |
+----------------------------------+------------------------------------------------------------------------------------------+
| *initial_dem_diff_pdf.png*       | Histogram of `initial_dh.tif`                                                            |
+----------------------------------+------------------------------------------------------------------------------------------+
| *final_dem_diff_pdf.png*         | Histogram of `final_dh.tif`                                                              |
+----------------------------------+------------------------------------------------------------------------------------------+


The files :

+----------------------+------------------------------------------------------------------------------------------+
| Name                 | Description                                                                              |
+======================+==========================================================================================+
| *final_config.json*  | | The configuration used for a particular demcompare run. It is the completion of the    |
|                      | | initial `config.json` file given by the user and contains additional information.      |
|                      | | It can be used to relaunch the same demcompare run or for a step by step run.          |
+----------------------+------------------------------------------------------------------------------------------+
| *stats_results-*     | | The files beginning with **'stats_results-'** are the `.json` and `.csv` files         |
|                      | | containing the statistics for each set. There is one file by mode.                     |
|                      | | See :ref:`statistics` for more details about the demcompare's statistics.              |
+----------------------+------------------------------------------------------------------------------------------+
| *histograms/*.npy*   | The **histograms/*.npy** files are the numpy histograms for each stats mode and set.     |
+----------------------+------------------------------------------------------------------------------------------+

And finally, the output sphinx report :ref:`report`.