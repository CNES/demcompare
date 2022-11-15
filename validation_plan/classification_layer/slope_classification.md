# Slope Classification:

`demcompare` handles classification layers to compute statistics on a specific group of pixels :
- They are superimposable to a dem 
- Slope layer classifies the dem slope 
- If a classification layer has two mask supports (ref and sec), then the ref support is used for the standard stats computation
- If a classification layer has two mask supports (ref and sec), then intersection and exclusion statistics are computed

# Existing tests:

### Tests SlopeClassification class (tests/classification_layer/test_slope_classification.py)

### `test_classify_slope_by_ranges`

**Test objective**

Test the classify_slope_by_ranges function
- Creates a slope dem
- Manually classifies the slope dem with the input ranges
- Tests that the classified slope by the function
  classify_slope_by_ranges is the same as ground truth

### `test_create_class_masks`

**Test objective**

Test the _create_class_masks function
- Creates a slope dem
- Manually classifies the slope dem with the input ranges
- Tests that the computed sets_masks_dict is equal to ground truth