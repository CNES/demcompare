# StatsDataset:

`demcompare` saves the computed statistics on a StatsDataset object:
- `StatsDataset` is a list of xarray dataset
- It creates an xarray dataset by classification layer
- Saves the statistics by class on the attrs
- Saves the statistics by class on mode intersection and exclusion on the attrs 
- Saves the input image 
- Saves the input image by class 
- Saves the input image by class on mode intersection and exclusion

# Existing tests:


### Tests StatsDataset class (tests/test_stats_dataset.py)

### `test_add_classif_layer_and_mode_stats_names`

**Test objective**

Test the add_classif_layer_and_mode_stats function.
Manually computes input stats for two classification
layers and different modes, and tests that the
add_classif_layer_and_mode_stats function correctly
adds the dataset names.


### `test_add_classif_layer_and_mode_stats_status_layer`

**Test objective**

Test the add_classif_layer_and_mode_stats function.
Manually computes input stats for two classification
layers and different modes, and tests that the
add_classif_layer_and_mode_stats function correctly
adds the Status layer information on the stats_dataset.

Also indirectly tests the get_dataset function.


### `test_add_classif_layer_and_mode_stats_slope_layer`

**Test objective**

Test the add_classif_layer_and_mode_stats function.
Manually computes input stats for two classification
layers and different modes, and tests that the
add_classif_layer_and_mode_stats function correctly
adds the Slope layer information on the stats_dataset.

Also indirectly tests the get_dataset function.



### `test_get_classification_layer_metric`

**Test objective**

Test the `get_classification_layer_metric` function.
    Manually computes input stats for one classification
    layer and different modes, and tests that the
    `get_classification_layer_metric` function correctly
    returns the corresponding metric value.

### `test_get_classification_layer_metrics`

**Test objective**

Test the `get_classification_layer_metrics` function.
    Manually computes input stats for classification
    layers and different modes, and tests that the
    `get_classification_layer_metrics` function correctly
    returns the corresponding metric names.

### `test_get_classification_layer_metrics_from_stats_processing`

**Test objective**

Tests the `get_classification_layer_metrics` function.
    Manually computes input stats for one classification
    layer and different modes, then computes more stats via the
    `StatsProcessing.compute_stats` API and tests that the
    `get_classification_layer_metrics` function correctly
    returns the metric names.

