# DatasetTools:

`demcompare` uses a dataset to store the input images:
- The demcompare dataset is a xarray dataset
- It stores the input image as a 2D dataarray
- It stores the classification layers as a 3D datarray
- It stores the slope as a 2D dataarray
- It stores the georef transform as a 1D dataarray

# Existing tests:

### Tests dataset_tools (tests/test_dataset_tools.py)

### `test_reproject_dataset`

**Test objective**

Test the `reproject_dataset` function.
    Loads the DEMS present in `strm_test_data` and `gironde_test_data`
    test root data directory and reprojects one
    onto another to test the obtained
    reprojected DEMs.


### `test_get_geoid_offset`

**Test objective**

Test the `_get_geoid_offset` function.
    Loads the DEMS present in `strm_test_data` test root data
    directory and projects it on the geoid to test
    the obtained dataset's geoid offset values.


### `test_get_geoid_offset_error`

**Test objective**

Test the `_get_geoid_offset` function.
Tests with an input transformation that will compute the data coordinates
    outside of the geoid scope and verifies that an error is raised.
