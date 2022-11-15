
# Fusion Classification:

`demcompare` handles classification layers to compute statistics on a specific group of pixels :
- They are superimposable to a dem 
- Fusion layer fusions at least two classification layers of any type
- If a classification layer has two mask supports (ref and sec), then the ref support is used for the standard stats computation
- If a classification layer has two mask supports (ref and sec), then intersection and exclusion statistics are computed

# Existing tests:

### Tests FusionClassification class (tests/classification_layer/test_fusion_classification.py)

### `test_create_merged_classes`

**Test objective**

Test the _create_merged_classes function
- Tests that the computed classes from _create_merged_classes
  are equal to ground truth

### `test_merge_classes_and_create_sets_masks`

**Test objective**

Test the _merge_classes_and_create_sets_masks function
- Creates a map image for both sec and ref supports
- Manually computes the standard, intersection and exclusion masks

### `test_create_labelled_map`

**Test objective**

Test the _create_labelled_map function
- Creates a map image for both sec and ref supports
- Tests that the computed map from _create_labelled_map
  is equal to ground truth

