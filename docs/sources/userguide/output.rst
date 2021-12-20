.. _outputs:

Outputs
=======



The stats
*********

The dh map and the intermediate data
************************************

`demcompare` will store several data and here is a brief explanation for each one.

First, the images :

- `intial_dh.tif` it the altitude differences image when both DEMs have been reprojected to the same **inputDSM** grid and no coregistration has been performed.

- `final_dh.tif` is the altitude differences image from the reprojected DEMs after the coregistration.

- the `dh_col_wise_wave_detection.tif` and `dh_row_wise_wave_detection.tif` are respectively computed by substituting
the `final_dh.tif` average col (row) to `final_dh.tif` itself. It helps to detect any residual oscillation.


- the `coreg_DSM.tif` and `coreg_Ref.tif` are the coregistered DEMS.

- the images whose names start with 'AltiErrors-' are the plots saved by `demcompare`. They show histograms by stats
set and the same histograms fitted by gaussian.

Then, the remaining files :

- the `final_config.json` is the configuration used for a particular demcompare run.
  - It is the completion of the initial `config.json` file given by the user and contains additional information.  It can be used to relaunch the same `demcompare` run or for a step by step run.

- the files beginning with **'stats_results-'** are the `.json` and `.csv` files containing the statistics for each set. There is one file by mode.

- the `histograms/*.npy` files are the numpy histograms for each stats mode and set.

Generated output documentation
******************************

The output `<test_output>/doc/published_report/` directory contains a full generated [sphinx](https://www.sphinx-doc.org/) documentation with all the results presented
for each mode and each set, in `html` or `latex` format.

It can be regenerated using `make html` or `make latexpdf` in `<test_output>/doc/src/` directory


