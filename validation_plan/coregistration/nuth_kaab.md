# Nuth et kaab coregistration algorithm:

`demcompare` can coregister two input DEMs using the nuth et kaab algorithm:
- It is a subclass of CoregistrationTemplate
- Takes two input dems, a reference one (ref) and a secondary one (sec)
- The two input dems must have the same resolution and size
- May save to disk the iteration plots
- The user must indicate the number of iterations

# Existing tests:

### Tests NuthKaab class (tests/coregistration/test_coregistration_nuth_kaab.py)


### `test_coregister_dems_algorithm_gironde_sampling_sec`

**Test objective**

Test the _coregister_dems_algorithm function of the Nuth & Kaab class.
    Loads the data present in the "gironde_test_data" root data
    directory and test that the output computed Transform is
    correct.

The following configurations are tested:
- "gironde_test_data" test root input DEMs, sampling value sec

### `test_coregister_dems_algorithm_gironde_sampling_ref`

**Test objective**

Test the _coregister_dems_algorithm function of the Nuth & Kaab class.
    Loads the data present in the "gironde_test_data" root data
    directory and test that the output computed Transform is
    correct.

The following configurations are tested:
- "gironde_test_data" test root input DEMs, sampling value ref

### `test_grad2d`

**Test objective**

Test the grad2d function
    Manually computes an input array and its
    slope and gradient, and tests that the obtained
    values resulting from the grad2d function are
    correct.

### `test_filter_target`

**Test objective**

Test the filter_target function
Computes an input target and manually adds noise
to it, then tests that the filter_target function
correctly filters the added noise.

### `test_nuth_kaab_single_iter`

**Test objective**

Manually computes an input array and its
output offsets, and tests that the resulting
offsets form the nuth_kaab_single_iter are the
same.

### `test_interpolate_dem_on_grid`

**Test objective**

Test the interpolate_dem_on_grid function
Manually computes an input array and its
spline interpolators, and tests that the resulting
splines form the interpolate_dem_on_grid are the
same.

### `test_crop_dem_with_offset_pos_x_pos_y` 

**Test objective**

Test the crop_dem_with_offset function with positive x and positive y offsets.
Manually computes an input array and crops it
with different offsets, and tests that the resulting
arrays form the crop_dem_with_offset are the
same.

### `test_crop_dem_with_offset_pos_x_neg_y` 

**Test objective**

Test the crop_dem_with_offset function with positive x and negative y offsets.
Manually computes an input array and crops it
with different offsets, and tests that the resulting
arrays form the crop_dem_with_offset are the
same.

### `test_crop_dem_with_offset_neg_x_pos_y` 

**Test objective**

Test the crop_dem_with_offset function with negative x and positive y offsets.

Manually computes an input array and crops it
with different offsets, and tests that the resulting
arrays form the crop_dem_with_offset are the
same.

### `test_crop_dem_with_offset_neg_x_neg_y` 

**Test objective**

Test the crop_dem_with_offset function with negative x and negative y offsets.
Manually computes an input array and crops it
with different offsets, and tests that the resulting
arrays form the crop_dem_with_offset are the
same.

# Not yet developed tests:

### `test_limit_iteration_number` 

**Test objective**

Test that the user can not specify an interation number higher
    than 15. 
