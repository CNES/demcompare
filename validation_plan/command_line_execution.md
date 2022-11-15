# Command line execution:

`demcompare`'s command line execution allows the user to:
- Write to disk the output coregistered dem
- Write to disk the output internal dems (reproj dem, reproj sec, reproj coreg dem, reproj coreg sec)
- Write to disk the output demcompare_results.json file with the obtained coregistration offsets
- Write to disk the coregistration algorithm outputs (ie. nuth et kaab plot iterations)
- Write to disk the statistics rectified maps by classification layer
- Write to disk the computed scalar metrics in a csv and json file by classification layer and mode
- Write to disk the computed vector metrics in a plot or a csv file by classification layer and mode

# Existing tests:


### Tests command line execution 

### Tests in tests/test_demcompare_gironde_data.py

### `test_demcompare_with_gironde_test_data`

**Test objective**

Demcompare with `gironde_test_data` main end2end test.
    Test that the outputs given by the Demcompare execution
    of `data/gironde_test_data/input/test_config.json` are
    the same as the reference ones
    in `data/gironde_test_data/ref_output/`

The tested files are: 
- `coregistration/coreg_SEC.tif`
- `coregistration/reproj_coreg_SEC.tif`
- `coregitration/reproj_coreg_REF.tif`
- `stats/Fusion0/sec_support_map_rectif.tif`
- `stats/Fusion0/stats_results.tif`
- `stats/global/stats_results.csv`
- `stats/Slope0/sec_support_map_rectif.tif`
- `stats/Slope0/ref_support_map_rectif.tif`
- `stats/Slope0/stats_results.csv`
- `stats/Slope0/stats_results_intersection.csv`
- `stats/Status/sec_support_map_rectif.tif`
- `stats/Status/stats_results.csv`
- `demcompare_results.json`
- `final_dem_diff.tif`
- `initial_dem_diff.tif`
- `test_config.json`

### `test_demcompare_with_gironde_test_data_sampling_ref`

**Test objective**

Demcompare with classification layer with
    sampling source ref main end2end test.
    Test that the outputs given by the Demcompare execution
    of `data/gironde_test_data_sampling_ref/input/test_config.json` are
    the same as the reference ones
    in `data/gironde_test_data_sampling_ref/ref_output/`


The tested files are: 
- `coregistration/coreg_SEC.tif`
- `coregistration/reproj_coreg_SEC.tif`
- `coregitration/reproj_coreg_REF.tif`
- `stats/Fusion0/ref_support_map_rectif.tif`
- `stats/Fusion0/stats_results.tif`
- `stats/global/stats_results.csv`
- `stats/Slope0/sec_support_map_rectif.tif`
- `stats/Slope0/ref_support_map_rectif.tif`
- `stats/Slope0/stats_results.csv`
- `stats/Slope0/stats_results_intersection.csv`
- `stats/Status/ref_support_map_rectif.tif`
- `stats/Status/stats_results.csv`
- `demcompare_results.json`
- `final_dem_diff.tif`
- `initial_dem_diff.tif`
- `test_config.json`

### Tests in tests/test_demcompare_strm_data.py

### `test_demcompare_strm_test_data`

**Test objective**

`strm_test_data` main end2end test.
    Test that the outputs given by the Demcompare execution
    of `data/strm_test_data/input/test_config.json`
    are the same as the reference ones
    in `data/strm_test_data/ref_output/`


The tested files are: 
- `coregistration/coreg_SEC.tif`
- `coregistration/reproj_coreg_SEC.tif`
- `coregitration/reproj_coreg_REF.tif`
- `snapshots/final_dem_diff_cdf.csv`
- `snapshots/final_dem_diff_pdf.csv`
- `snapshots/initial_dem_diff_cdf.csv`
- `snapshots/inital_dem_diff_pdf.csv`
- `stats/Slope0/stats_results.csv`
- `stats/Slope0/stats_results_intersection.csv`
- `stats/Slope0/stats_results_exclusion.csv`
- `demcompare_results.json`
- `final_dem_diff.tif`
- `initial_dem_diff.tif`
- `test_config.json`


### `test_demcompare_strm_test_data_with_roi`

**Test objective**

`strm_test_data_with_roi` main end2end test with ROI input.
    Test that the outputs given by the Demcompare execution
    of `data/strm_test_data_with_roi/input/test_config.json` are the same
    as the reference ones in `data/strm_test_data_with_roi/ref_output/`

The tested files are: 
- `snapshots/final_dem_diff_cdf.csv`
- `snapshots/final_dem_diff_pdf.csv`
- `snapshots/initial_dem_diff_cdf.csv`
- `snapshots/inital_dem_diff_pdf.csv`
- `stats/Slope0/stats_results.csv`
- `stats/Slope0/stats_results_intersection.csv`
- `demcompare_results.json`
- `final_dem_diff.tif`
- `initial_dem_diff.tif`
- `test_config.json`

### Tests in tests/test_demcompare_steps.py

### `test_demcompare_coregistration_step_with_gironde_test_data`

**Test objective**

Demcompare with gironde_test_data coregistration test.
    Test that the outputs given by the Demcompare execution
    of only the coregistration step in `data/gironde_test_data/input/test_config.json` are
    the same as the reference ones
    in `data/gironde_test_data/ref_output/`

The tested files are: 
- `demcompare_results.json`

### `test_demcompare_statistics_step_with_gironde_test_data`

**Test objective**

Demcompare with `gironde_test_data` running directly the
    statistics step test with both input_ref and input_sec as inputs.
    Test that the cfg is executed without errors.

### `test_demcompare_statistics_step_input_ref_with_gironde_test_data`

**Test objective**

Demcompare with `gironde_test_data` running directly the
    statistics step test with a single input dem.
    Test that the `cfg` is executed without errors.

# Not yet developed tests:

### `test_demcompare_with_output_cfg_file` 

**Test objective**

Test that demcompare can be directly executed with a previous execution's 
    output configuration file. 

### `test_demcompare_with_two_equal_input_dems` 

**Test objective**

Test that the outputs are clear enough when two equal input dems are
    given as `input_ref` and `input_sec`. 
- If the coregistration is to be computed: according to the coregistration algorithm, 
  verify if the output should be zero coregistration offsets, or an output error such as: 
  "Coregistration error, not enough alti_diff values to compute coregistration offsets" should be raised. 
- If the statistics are to be computed: test that the metrics are correctly computed on a 
  zero `alti_diff` dem. 

### `test_demcompare_with_one_empty_input_dem` 

**Test objective**

Test that demcompare gives the following error when one of the input dems is empty (all NaNs):
    "Input error, ref/sec dem has only invalid values."


### `test_demcompare_with_voluminous_data` 

**Test objective**

Test that demcompare correctly proceeds to the cropping of the input dems when they are
    detected as being too big to handle. 

### `test_demcompare_exceptions` 

**Test objective**

Test that demcompare correctly handles exeptions from the different error tests cases.
Ensure that exceptions are handled from the main level.
Exceptions that should be handled with examples on each case (those examples do not cover all possible exeptions for each case): 

- Invalid input dems (i.e. dem with all invalid values)
- Invalid input classification map (i.e. different size than support dem)
- Invalid input dems arguments (i.e. non-existing alti unit)
- Invalid coregistration arguments (i.e. non-existing method, invalid number of iterations)
- Invalid statistics arguments (i.e. non-existing metric, non-existing argument)
- Invalid statistics classification layers arguments (i.e. non-existing layer type, segmentation layer with input ranges, fusion layer with input classes)
- Reprojection: not being able to be performed (i.e. not common ROI between both dems)
- Coregistration: not being able to be performed (i.e. not enough values to perform the estimation)
- Statistics: not being able to create slope layer (i.e. slope not previously computed)
- Statistics: not being able to create fusion layer (i.e. layers to be fused are not defined on the specified support dem)
- Statistics: not being able to create segmentation layer (i.e. support map is not present on input dem)
- Statistics API: not being able to perform the desired API function (i.e. compute/retrieve non-existing metric, compute/retrieve on non-existing classification layer, compute/retrieve on non-existing mode, compute/retrieve on non-existing class)