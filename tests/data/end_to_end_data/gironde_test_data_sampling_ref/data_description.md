## Data description

The test data **classification_layer_sampling_ref** folder contains the following elements:

* **input** folder containing:
  * A reference DEM *Gironde.tif* of size 1093x1142 pixels and resolution (-0.0010416, 0.0010416).
  * A dem to align *FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF.tif* of size 218x218 pixels and resolution (500.00, -500.00).
    * A classification layer map named **Status** for the sec**FinalWaveBathymetry_T30TXR_20200622T105631_Status.TIF* of size 218x218 pixels and resolution (500.00, -500.00).
  * *test_config.json* : input configuration file to run demcompare on the input dems with *nuth_kaab_internal* and a *sampling_source = ref*.
* **ref_output** folder containing:
  * *test_config.json* : resulting input configuration file from running demcompare (filled with the defaut parameters when not set).
  * *demcompare_results.json*: output results from coregistration and stats.
  * *final_dh.tif* and *initial_dh.tif*: initial and final altitude difference DEMs to evaluate the coregistration.
  * **coregistration** folder: internal DEMs of the coregistration and output coregistered dem to align (*coreg_SEC.tif*).
  * **stats**: one folder for each classification layer (**slope**, **Status** and **fusion_layer**) containing the *.csv* files of the *exclusion/intersection* segmentations and the *support_map.tif* if it is to be tested.
