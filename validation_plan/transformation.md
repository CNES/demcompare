# Transformation:

`demcompare`'s coregistration gives a transformation object as output:
- It contains the `x_offset`, `y_offset` and `z_offset`
- It can apply the `x_offset` and `y_offset` to an input dem

# Existing tests:

### Tests transformation (tests/test_transformation.py)

### `test_apply`

**Test objective**

Test the `apply_transform` function
- Creates a DEM xr.Dataset with the georefence
    from the `strm_test_data` test root data directory
- Creates a `Transform` object. 
- Applies the transform to
    the created DEM to test that the transformation has been
    correctly applied.

### `test_adapt_transform_offset` 

**Test objective**

Test the `adapt_transform_offset` function
- Creates a `Transform` object and an adapting factor
- Tests that the offsets has been correctly adapted by the input `adapting_factor`.


# Not yet developed tests:

### `test_apply_original_dem` 

**Test objective**

Test that the dem given to the `transformation.apply` does not have its
    `georeference_transform` modified, only the returned dem does. 

