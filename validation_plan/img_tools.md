# Img_tools:

`demcompare`'s handles raster image fuctions, mainly consisting on rasterio wrappers

# Existing tests:


### `test_convert_pix_to_coord_neg_x_pos_y`

**Test objective**

Test `convert_pix_to_coord` function.
Makes the conversion from pix to coord with negative x and positive y.
Makes the conversion from pix to coord for
different pixel values and tests the obtained
coordinates.

### `test_convert_pix_to_coord_pos_x_pos_y`

**Test objective**

Test `convert_pix_to_coord` function.
Makes the conversion from pix to coord with positive x and positive y.
Makes the conversion from pix to coord for
different pixel values and tests the obtained
coordinates.

### `test_convert_pix_to_coord_neg_x_neg_y`

**Test objective**

Test `convert_pix_to_coord` function.
Makes the conversion from pix to coord with negative x and negative y.
Makes the conversion from pix to coord for
different pixel values and tests the obtained
coordinates.

### `test_convert_pix_to_coord_pos_x_neg_y`

**Test objective**

Test `convert_pix_to_coord` function.
Makes the conversion from pix to coord with positive x and negative y.
Makes the conversion from pix to coord for
different pixel values and tests the obtained
coordinates.

### `test_compute_gdal_translate_bounds`

**Test objective**

Test the `compute_offset_bounds` function.
    Loads the DEMS present in `strm_test_data` and `gironde_test_data"`
    test root data directory and computes the coordinate offset
    bounds for a given pixellic offset to test the resulting
    bounds.
