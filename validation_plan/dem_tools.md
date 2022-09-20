# INPUT DEMs:

`demcompare` supports input DEMs:
- If one single dem is given, it must be named "input_ref"
- If two dems are given, they must be named "input_ref" and "input_sec"
- The input dems may have different `z` units
- The input dems may have different resolution
- The input dems may have different coordinate systems
- The input dems may have different nodata values
- The input dems may have an input ROI
- The input dems may have different `z` origin (geoid or ellipsoid)


### Tests dem_tools (tests/test_dem_tools.py)

### `test_load_dem`

**Test objective**

Test the load_dem function.
    Loads the data present in `strm_test_data` test root
    data directory and tests the loaded DEM dataset.

Loaded parameters being tested: 
- `nodata`
- `xres`
- `yres`
- `plani_unit`
- `zunit`
- `georef`
- `shape`
- `transform`

# Existing tests:


### Tests dem_tools (tests/test_dem_tools.py)

### `test_translate_dem_pos_x_neg_y`

**Test objective**

Test the `translate_dem` function with positive x
and negative y offsets.

Loads the DEMS present in `gironde_test_data` test root
data directory and makes the DEM translation for
different pixel values and tests the resulting
transform.

### `test_translate_dem_neg_x_neg_y`

**Test objective**

Test the `translate_dem` function with negative x
    and negative y offsets.

### `test_translate_dem_neg_x_pos_y`

**Test objective**

Test the `translate_dem` function with negative x
    and positive y offsets.

### `test_translate_dem_pos_x_pos_y`

**Test objective**

Test the `translate_dem` function with positive x
    and positive y offsets.

### `test_translate_dem_no_offset`

**Test objective**

Test the `translate_dem` function without input offset.


### `test_reproject_dems_sampling_sec`

**Test objective**

Test the `reproject_dems` function.
    Loads the DEMS present in `gironde_test_data` test root
    data directory and reprojects them to test the
    obtained reprojected DEMs.

Test configuration: 
- Reproject dems with sampling value dem

### `test_reproject_dems_sampling_ref`

**Test objective**

Test the `reproject_dems` function.
    Loads the DEMS present in `gironde_test_data` test root
    data directory and reprojects them to test the
    obtained reprojected DEMs.

Test configuration: 
- Reproject dems with sampling value ref

### `test_reproject_dems_sampling_sec_initial_disparity`

**Test objective**

Test the `reproject_dems` function.
    Loads the DEMS present in `gironde_test_data` test root
    data directory and reprojects them to test the
    obtained reprojected DEMs.

Test configuration: 
- Reproject dems with sampling value dem and initial disparity

### `test_compute_dems_diff`

**Test objective**

Test `compute_dems_diff `function.
    Creates two DEM datasets and computes their altitude differences
    to test the obtained difference Dataset.

### `test_create_dem`

**Test objective**

Test `the _create_dem` function.
    Creates a dem with the data present in
    `gironde_test_data` test root data directory and tests
    the obtained DEM Dataset.

Test configuration: 
- Test with geoid_georef set to False
- Test with geoid_georef set to True

### `test_create_dem_with_classification_layers_dictionary`

**Test objective**

Test the `_create_dem` function with input classification layers
    Creates a dem with random input data and input classification layers
    as xr.DataArray and as a dictionary.

Test configuration: 
- Test with input classification layer as a dictionary

### `test_create_dem_with_classification_layers_dataarray`

**Test objective**

Test the `_create_dem` function with input classification layers
    Creates a dem with random input data and input classification layers
    as xr.DataArray and as a dictionary.

Test configuration: 
- Test with input classification layer as an xr.DataArray

### `test_compute_waveform`

*Test objective*

Test the `compute_waveform` function.

### `test_compute_dem_slope`

**Test objective**

Test the `compute_dem_slope` function.
- Loads the data present in the test root data directory
- Creates a dem with a created array and the input data
  georeference and transform
- Manually computes the dem slope
- Tests that the computed slope by the function
  compute_dem_slope is the same as ground truth

### `test_verify_fusion_layers_sec`

**Test objective**

Test the `verify_fusion_layers` function.
- Loads the data present in the test root data directory
- Manually computes different classification layers configuration
  that include fusion layers
- Tests that the verify_fusion_layers raises an error when
  the layers needed to be fused are not present on the input dem
  and the input cfg

Test configuration: 
- Test with correct sec fusion

### `test_verify_fusion_layers_error_ref`

**Test objective**

Test the `verify_fusion_layers` function.
- Loads the data present in the test root data directory
- Manually computes different classification layers configuration
  that include fusion layers
- Tests that the verify_fusion_layers raises an error when
  the layers needed to be fused are not present on the input dem
  and the input cfg

Test configuration: 
- Test with a degraded ref fusion where one of the specified layers to be fused does not exist
    on the ref dem 


### `test_verify_fusion_layers_cfg_error`

**Test objective**

Test the `verify_fusion_layers` function.
- Loads the data present in the test root data directory
- Manually computes different classification layers configuration
  that include fusion layers
- Tests that the verify_fusion_layers raises an error when
  the layers needed to be fused are not present on the input dem
  and the input cfg

Test configuration:
- Test with a degraded ref fusion where one of the specified layers to be fused was not
    defined in the input cfg.


# Not yet developed tests:

### `test_load_and_reproject_different_z_units` 

**Test objective**

Test that two dems loaded with different alti units (ie. one in `cm` and another in `m`)
    are correctly loaded and reprojected. 


### `test_classification_layer_mask_with_wrong_size` 

**Test objective**

Test that decompare's `load_dems` function raises an error when the 
    input classification layer mask of the dem has a different size
    than its support dem


### `test_reproject_dems_without_intersection` 

**Test objective**

Test that demcompare's `reproject_dems` function raises an error when the
    input dems do not have a common intersection. 


### `test_translate_dem_original_dem` 

**Test objective**

Test that the dem given to the translate funcion does not have its
    `georeference_transform` modified, only the returned dem does. 


### `test_wrong_classification_map` 

**Test objective**

Test that the `load_dem` function raises an error when given a classification layer map path
that has different dimensions than its support. 
