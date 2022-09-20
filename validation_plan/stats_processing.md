# StatsProcessing:

`demcompare` computes statistics:
- On a difference between two dems
- On an input dem 
- With a pixel classification by class
- With a fusion of classifications by class
- If two supports are available on the classification (ref and sec), classifying 
  the pixels belonging to the same class for both supports
- Giving as output a `StatsDataset`
- Optionally saving to disk the output csv and json files


# Existing tests:


### Tests StatsProcessing class (tests/test_stats_processing.py)

### `test_create_classif_layers`

**Test objective**

Test the `create_classif_layers` function
    Creates a `StatsProcessing` object with an input configuration
    and tests that its created classification layers
    are the same as gt.

 
### `test_create_classif_layers_without_input_classif`

**Test objective**

Test the `create_classif_layers` function
    Creates a `StatsProcessing` object with an input configuration
    that does not specify any classification layer
    and tests that its created classification layers
    are the same as the default gt.


### `test_compute_stats_slope_layer`

**Test objective**

Tests the `compute_stats`. Manually computes
    the stats for different `classification_layer_masks` for a given class
    and mode and tests that the `compute_stats` function obtains
    the same values in the same values on the slope layer.


### `test_compute_stats_global_layer`

**Test objective**

Tests the `compute_stats`. Manually computes
    the stats for different `classification_layer_masks` for a given class
    and mode and tests that the `compute_stats` function obtains
    the same values in the same values on the global layer.


### `test_compute_stats_segmentation_layer`

**Test objective**

Tests the `compute_stats`. Manually computes
    the stats for different `classification_layer_masks` for a given class
    and mode and tests that the `compute_stats` function obtains
    the same values in the same values on the segmentation layer.

### `test_compute_stats_slope_classif_intersection_mode`

**Test objective**

Tests the `compute_stats` for `intersection` mode. Manually computes
    the stats for the slope function for a given class
    and `intersection` mode using the `stats_processing` API and tests that the
    `compute_stats` function obtains
    the same values.


### `test_compute_stats_slope_classif_exclusion_mode`

**Test objective**

Tests the `compute_stats` for `exclusion` mode. Manually computes
    the stats for the slope function for a given class
    and `exclusion` mode using the `stats_processing` API and tests that the
    `compute_stats` function obtains
    the same values.


### `test_compute_stats_from_cfg_status`

**Test objective**

Tests the `compute_stats` function with different metrics specified in the cfg for different classification layers. Manually computes
    the stats for the `classification_layer_masks` and tests that the compute_stats function obtains
    the same values for the status layer.


### `test_compute_stats_from_cfg_slope`

**Test objective**

Tests the `compute_stats` function with different metrics specified in the cfg for different classification layers. Manually computes
    the stats for the `classification_layer_masks` and tests that the compute_stats function obtains
    the same values for the slope layer.

# Not yet developed tests:

### `test_statistics_save_results` 

**Test objective**

Test that demcompare's execution with the statistics `save_results` parameter
    set to `True` correctly saves to disk all classification layer's maps, csv and json files. 
Test that demcompare's execution with the statistics `save_results` parameter
    set to `False` does not save to disk the classification layer's maps, csv or json files.
