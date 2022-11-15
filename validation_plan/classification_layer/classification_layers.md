
# Input Classification:

`demcompare` handles classification layers to compute statistics on a specific group of pixels :
- They are superimposable to a dem 
- They can be of type global, segmentation, slope or fusion
- Global layer is always computed, it has a single class containing all valid pixels
- Segmentation layer needs an input classification mask as input
- Slope layer classifies the dem slope 
- Fusion layer fusions at least two classification layers of any type
- If a classification layer has two mask supports (ref and sec), then the ref support is used for the standard stats computation
- If a classification layer has two mask supports (ref and sec), then intersection and exclusion statistics are computed


# Existing tests:

### Tests classification_layer class (tests/classification_layer/test_classification_layer.py)


### `test_get_outliers_free_mask`

**Test objective**

Test the _get_outliers_free_mask function
- Manually computes an input array and filters it
- Tests that the resulting arrays form the _get_outliers_free_mask are the same.

### `test_get_nonan_mask_defaut_nodata`

**Test objective**

Test the _get_nonan_mask function with default nodata value.
- Manually computes an input array and filters it
- Tests that the resulting arrays form the _get_nonan_mask are the same.

### `test_get_nonan_mask_custom_nodata`

**Test objective**

Test the _get_nonan_mask function with custom nodata value.
- Manually computes an input array and filters it
- Tests that the resulting arrays form the _get_nonan_mask are the same.
- 
### `test_create_mode_masks`

**Test objective**

Test the _create_mode_masks function  
- Creates a map image for both sec and ref supports
- Manually computes the standard, intersection and exclusion masks
- Tests that the computed masks from _create_mode_masks are equal to ground truth


# Not yet developed tests:

### `test_statistics_classification_invalid_input_classes` 

**Test objective**

Test that demcompare's initialization fails when the configuration file
    specifies a classification layer with invalid class values. The error
    should be raised in advance, before the coregistration step if it is present. 

### `test_statistics_classification_invalid_input_ranges` 

**Test objective**

Test that demcompare's initialization fails when the configuration file
    specifies a classification layer with invalid range values. The error
    should be raised in advance, before the coregistration step if it is present. 

### `test_demcompare_with_wrong_fusion_cfg` 

**Test objective**

Test that demcompare's initialization fails when the configuration file
    specifies a fusion layer with a single layer to be fused. The error
    should be raised in advance, before the coregistration step if it is present.



