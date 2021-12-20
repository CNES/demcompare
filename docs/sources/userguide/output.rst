.. _outputs:

Outputs
=======


Demcompare will store several data and here is a brief explanation for each one.

First, the images :

- **intial_dh.tif** it the altitude differences image when both DEMs have been reprojected to the same inputDSM grid and no coregistration has been performed.

- **final_dh.tif** is the altitude differences image from the reprojected DEMs after the coregistration.

- The **dh_col_wise_wave_detection.tif** and **dh_row_wise_wave_detection.tif** are respectively computed by substituting the `final_dh.tif` average col (row) to `final_dh.tif` itself. It helps to detect any residual oscillation.


- The **coreg_DSM.tif** and **coreg_Ref.tif** are the coregistered DEMS.

  - As explained on :ref:`coregistration`, both coregistered DEMs will have the **dsm**'s georeference grid and the **ref**'s georeference origin.

- The images whose names start with **AltiErrors-** are the plots saved by Demcompare. They show histograms by stats set and the same histograms fitted by gaussian.

Then, the remaining files :

- The **final_config.json** is the configuration used for a particular Demcompare run.

  - It is the completion of the initial `config.json` file given by the user and contains additional information.  It can be used to relaunch the same Demcompare run or for a step by step run.

- The files beginning with **'stats_results-'** are the `.json` and `.csv` files containing the statistics for each set. There is one file by mode.

  - See :ref:`statistics` for more details about the Demcompare's statistics.

- The **histograms/*.npy** files are the numpy histograms for each stats mode and set.

- The output sphinx report :ref:`report`