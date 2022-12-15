# Command line execution:

`demcompare`'s command line execution allows the user to:
- Write to disk the output coregistered dem
- Write to disk the output internal dems (reproj dem, reproj sec, reproj coreg dem, reproj coreg sec)
- Write to disk the output demcompare_results.json file with the obtained coregistration offsets
- Write to disk the coregistration algorithm outputs (ie. nuth et kaab plot iterations)
- Write to disk the statistics rectified maps by classification layer
- Write to disk the computed scalar metrics in a csv and json file by classification layer and mode
- Write to disk the computed vector metrics in a plot or a csv file by classification layer and mode

# Not yet developed tests:

### `test_demcompare_with_output_cfg_file` 

**Test objective**

Test that demcompare can be directly executed with a previous execution's 
    output configuration file. 

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