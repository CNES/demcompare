
# Segmentation Classification:

`demcompare` handles classification layers to compute statistics on a specific group of pixels :
- They are superimposable to a dem 
- Segmentation layer needs an input classification mask as input
- If a classification layer has two mask supports (ref and sec), then the ref support is used for the standard stats computation
- If a classification layer has two mask supports (ref and sec), then intersection and exclusion statistics are computed

# Existing tests:

### Tests SegmentationClassification class (tests/classification_layer/test_segmentation_classification.py)

### `test_create_labelled_map`

**Test objective**

Test the `_create_labelled_map` function
Manually computes an input dem with two input
classification layers,
then creates the classification layer object
and verifies the computed map_image (function `_create_labelled_map`)

### `__test_create_class_masks`

**Test objective**

Test the `_create_classification_layer_class_masks` function
Manually computes an input dem and with two input
classification layers,
then creates the first classification layer object
and verifies the computed sets_masks_dict
(function `_create_class_masks`)
