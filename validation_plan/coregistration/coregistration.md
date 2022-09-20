# Coregistration:

`demcompare` can coregister two input DEMs:
- Using the Nuth & Kaab algorithm
- Taking two input dems, a reference one (ref) and a secondary one (sec)
- Performing a reprojection to make both dems have the same resolution and size
- Letting the user choose between `ref` and `sec` working resolutions (default is the sec one)
- Applying the obtained offsets to the sec DEM
- Optionally writing on disk the coregistered sec DEM 
- Optionally writing on disk the reprojected dems and the coregistered reprojected dems


# Existing tests:

### Tests Coregistration class (tests/coregistration/test_coregistration.py)

### Tests compute_coregistration_function 

Four differents tests with the same test objective and different input configurations
are available. 


**Test objective**

Test the compute_coregistration function:
- Loads the data from the test root data directory
- Creates a coregistration object and does compute_coregistration
- Tests that the output Transform has the correct offsets
- Tests that the considered sampling source is correct
- Test that the offsets, bias and gdal_translate_bounds
  on the demcompare_results output dict are corrects

### `test_compute_coregistration_with_gironde_test_data_sampling_dem`

Test configuration:
- "gironde_test_data" input DEMs
- sampling value dem
- coregistration method Nuth & kaab

### `test_compute_coregistration_with_gironde_test_data_sampling_ref` 

Test configuration:
- "gironde_test_data" input DEMs
- sampling value ref
- coregistration method Nuth & kaab


## `test_compute_coregistration_with_strm_sampling_dem_and_initial_disparity` 

Test configuration:
- "strm_test_data" input DEMs
- sampling value ref
- coregistration method Nuth & kaab
- non-zero initial disparity


### `test_compute_coregistration_gironde_sampling_dem_and_initial_disparity` 

Test configuration:
- "gironde_test_data" input DEMs
- sampling value dem
- coregistration method Nuth & kaab
- non-zero initial disparity

### `test_compute_coregistration_gironde_sampling_ref_and_initial_disparity` 

Test configuration:
- "gironde_test_data" input DEMs
- sampling value ref
- coregistration method Nuth & kaab
- non-zero initial disparity

### `test_compute_coregistration_with_default_coregistration_strm_sampling_dem`

Test configuration:
- default coregistration without input configuration
- "strm_test_data" input DEMs
- sampling value dem
- coregistration method Nuth & kaab
- non-zero initial disparity



# Not yet developed tests:


### `test_coregistration_save_internal_dems` 

**Test objective**

Test that demcompare's execution with the coregistration save_internal_dems parameter
    set to True correctly saves the dems to disk: 
- reproj_ref.tif
- reproj_sec.tif
- reproj_coreg_ref.tif
- reproj_coreg_sec.tif
Test that demcompare's execution with the coregistration save_internal_dems parameter
set to False does not save to disk the previous listed dems. 

### `test_coregistration_save_coreg_method_outputs` 

**Test objective**

Test that demcompare's execution with the coregistration save_coreg_method_outputs parameter
    set to True correctly saves to disk the iteration plots of Nuth et kaab. 
Test that demcompare's execution with the coregistration save_coreg_method_outputs parameter
    set to False does not save to disk the iteration plots of Nuth et kaab.  

### `test_coregistration_with_output_dir` 

**Test objective**

Test that demcompare's execution with the output_dir being specified correctly
    saves to disk the dem coreg_sec.tif and the output file demcompare_results.json
Test that demcompare's execution with the output_dir not being specified and
    the parameters save_internal_dems and/or save_coreg_method_outputs set to True
    does rise an error. 


### `test_coregistration_with_wrong_initial_disparities` 

**Test objective**

Test that demcompare's initialization fails when the coregistration specifies
    an invalid initial disparity value. 
