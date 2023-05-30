## Data description

The test data **strm_test_data** folder contains the following elements:

* **input** folder containing:
  * A reference DEM *strm_ref.tif* of size 1000x1000 pixels.
  * A dem to align *strm_blurred_and_shifted.tif* that has been manually created by blurring and shifting the reference DEM by (3, 5) pixels. Its size is 997x995 pixels.
    * Both reference and dem to align have the same resolution.
  * *test_config.json* : input configuration file to run demcompare on the input dems with *nuth_kaab_internal* and the default *sampling_source* (*sec*).
* **ref_output** folder containing:
  * *test_config.json* : resulting input configuration file from running demcompare (filled with the defaut parameters when not set).
  * *demcompare_results.json*: output results from coregistration and stats.
  * *final_dh.tif* and *initial_dh.tif*: initial and final altitude difference DEMs to evaluate the coregistration.
  * **coregistration** folder: internal DEMs of the coregistration and output coregistered dem to align (*coreg_SEC.tif*).
  * **stats**: contains:
    * one folder for each classification layer containing the *.csv* files of the *exclusion/intersection segmentations and the*support_map.tif* if it is to be tested.
    * initial and final *.csv* files of the PDF (Probability Density Function) and CDF (Cumulative Density Function) to evaluate the coregistration.
