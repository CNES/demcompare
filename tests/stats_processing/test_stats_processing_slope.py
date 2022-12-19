#!/usr/bin/env python
# coding: utf8
# Disable the protected-access to test the functions
# pylint:disable=protected-access
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of demcompare
# (see https://github.com/CNES/demcompare).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This module contains functions to test the
methods for slope layer in the StatsProcessing class.
"""

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare.metric import Metric

# pylint:disable = duplicate-code


@pytest.mark.unit_tests
@pytest.mark.functional_tests
def test_compute_stats_slope_layer(initialize_stats_processing):
    """
    Tests the compute_stats for a slope classification layer
    Input data:
    - StatsProcessing object from the "initialize_stats_processing"
      fixture
    Validation data:
    - The manually computed statistics on the pixels
      corresponding to the class 0 of the slope classification
      layer: gt_mean, gt_sum,
      gt_ratio
    Validation process:
    - Creates the StatsProcessing object
    - Computes the nodata_nan_mask of the StatsProcessing's
      input dem
    - Computes the ground truth metrics on the valid pixels
      corresponding to the class 0
    - Computes the metrics using the StatsProcessing's compute_stats
      function and gets the result using the get_classification_layer_metric
      function
    - Checks that the obtained metrics are the same as ground truth
    - Checked function: StatsProcessing's
      compute_stats and get_classification_layer_metric
    """
    # Initialize stats processing object
    stats_processing = initialize_stats_processing

    # Test slope layer class 0
    # ----------------------------------------------------------
    classif_class = 0
    dataset_idx = stats_processing.classification_layers_names.index("Slope0")

    # Compute nodata_nan_mask
    nodata_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != nodata_value),
        0,
        stats_processing.dem["image"].data,
    )
    # Get valid class indexes for slope dataset and class 0
    idxes_map_class_0_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[
            "ref"
        ][classif_class]
    )
    # Get class pixels
    classif_map = stats_processing.dem["image"].data[
        idxes_map_class_0_dataset_0
    ]

    # Initialize metric objects
    metric_mean = Metric("mean")
    metric_sum = Metric("sum")
    # Compute gt metrics
    gt_mean = metric_mean.compute_metric(classif_map)
    gt_sum = metric_sum.compute_metric(classif_map)
    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(
        classification_layer=["Slope0"], metrics=["mean", "sum"]
    )
    # Get output metrics
    output_mean = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="standard",
        metric="mean",
    )
    output_sum = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="standard",
        metric="sum",
    )
    # Test that the output metrics are the same as gt
    np.testing.assert_allclose(
        gt_mean,
        output_mean,
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_sum,
        output_sum,
        rtol=1e-02,
    )
    # Do the same test with the ratio_above_threshold 2D metric
    # Initialize metric objects
    elevation_thrlds = [-3, 2, 90]
    metric_ratio = Metric(
        "ratio_above_threshold",
        params={"elevation_threshold": elevation_thrlds},
    )
    # Compute gt metrics
    gt_ratio = metric_ratio.compute_metric(classif_map)
    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(
        classification_layer=["Slope0"],
        metrics=[
            {"ratio_above_threshold": {"elevation_threshold": elevation_thrlds}}
        ],
    )
    # Get output metrics
    output_ratio = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="standard",
        metric="ratio_above_threshold",
    )
    # Test that the output metrics are the same as gt
    np.testing.assert_allclose(
        gt_ratio,
        output_ratio,
        rtol=1e-02,
    )


@pytest.mark.unit_tests
@pytest.mark.functional_tests
def test_compute_stats_slope_classif_intersection_mode(
    initialize_stats_processing,
):
    """
    Tests the compute_stats for a slope classification layer and
    intersection mode
    Input data:
    - StatsProcessing object from the "initialize_stats_processing"
      fixture
    Validation data:
    - The manually computed statistics on the pixels
      corresponding to the class 0 of the slope classification
      layer in mode intersection: gt_mean, gt_sum,
      gt_ratio
    Validation process:
    - Creates the StatsProcessing object
    - Computes the nodata_nan_mask of the StatsProcessing's
      input dem
    - Computes the ground truth metrics on the valid pixels
      corresponding to the class 0 in mode intersection
    - Computes the metrics using the StatsProcessing's compute_stats
      function and gets the result using the get_classification_layer_metric
      function
    - Checks that the obtained metrics are the same as ground truth
    - Checked function: StatsProcessing's
      compute_stats and get_classification_layer_metric
    """
    # Initialize stats processing object
    stats_processing = initialize_stats_processing

    # Test slope layer class 0 mode intersection
    # ----------------------------------------------------------
    classif_class = 0
    dataset_idx = stats_processing.classification_layers_names.index("Slope0")

    # Compute nodata_nan_mask
    nodata_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != nodata_value),
        0,
        stats_processing.dem["image"].data,
    )

    # Get valid class indexes for slope dataset and class 0 mode intersection
    idxes_map_class_0_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[
            "ref"
        ][classif_class]
        * stats_processing.classification_layers[dataset_idx].classes_masks[
            "sec"
        ][classif_class]
    )
    # Get class pixels
    classif_map = stats_processing.dem["image"].data[
        idxes_map_class_0_dataset_0
    ]

    # Initialize metric objects
    metric_mean = Metric("mean")
    metric_sum = Metric("sum")
    # Compute gt metrics
    gt_mean = metric_mean.compute_metric(classif_map)
    gt_sum = metric_sum.compute_metric(classif_map)
    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(
        classification_layer=["Slope0"], metrics=["mean", "sum"]
    )
    # Get output metrics
    output_mean = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="intersection",
        metric="mean",
    )
    output_sum = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="intersection",
        metric="sum",
    )
    # Test that the output metrics are the same as gt
    np.testing.assert_allclose(
        gt_mean,
        output_mean,
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_sum,
        output_sum,
        rtol=1e-02,
    )
    # Do the same test with the ratio_above_threshold 2D metric
    # Initialize metric objects
    elevation_thrlds = [-3, 2, 90]
    metric_ratio = Metric(
        "ratio_above_threshold",
        params={"elevation_threshold": elevation_thrlds},
    )
    # Compute gt metrics
    gt_ratio = metric_ratio.compute_metric(classif_map)
    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(
        classification_layer=["Slope0"],
        metrics=[
            {"ratio_above_threshold": {"elevation_threshold": elevation_thrlds}}
        ],
    )
    # Get output metrics
    output_ratio = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="intersection",
        metric="ratio_above_threshold",
    )
    # Test that the output metrics are the same as gt
    np.testing.assert_allclose(
        gt_ratio,
        output_ratio,
        rtol=1e-02,
    )


@pytest.mark.unit_tests
@pytest.mark.functional_tests
def test_compute_stats_slope_classif_exclusion_mode(
    initialize_stats_processing,
):
    """
    Tests the compute_stats for a slope classification layer and
    exclusion mode
    Input data:
    - StatsProcessing object from the "initialize_stats_processing"
      fixture
    Validation data:
    - The manually computed statistics on the pixels
      corresponding to the class 0 of the slope classification
      layer in mode exclusion: gt_mean, gt_sum,
      gt_ratio
    Validation process:
    - Creates the StatsProcessing object
    - Computes the nodata_nan_mask of the StatsProcessing's
      input dem
    - Computes the ground truth metrics on the valid pixels
      corresponding to the class 0 in mode exclusion
    - Computes the metrics using the StatsProcessing's compute_stats
      function and gets the result using the get_classification_layer_metric
      function
    - Checks that the obtained metrics are the same as ground truth
    - Checked function: StatsProcessing's
      compute_stats and get_classification_layer_metric
    """
    # Initialize stats processing object
    stats_processing = initialize_stats_processing

    # Test slope layer class 0
    # ----------------------------------------------------------
    classif_class = 0
    dataset_idx = stats_processing.classification_layers_names.index("Slope0")

    # Compute nodata_nan_mask
    nodata_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != nodata_value),
        0,
        stats_processing.dem["image"].data,
    )

    # Get valid class indexes for slope dataset and class 0 mode exclusion
    idxes_map_class_0_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[
            "ref"
        ][classif_class]
        * (
            1
            - (
                stats_processing.classification_layers[
                    dataset_idx
                ].classes_masks["ref"][classif_class]
                * stats_processing.classification_layers[
                    dataset_idx
                ].classes_masks["sec"][classif_class]
            )
        )
    )
    # Get class pixels
    classif_map = stats_processing.dem["image"].data[
        idxes_map_class_0_dataset_0
    ]

    # Initialize metric objects
    metric_mean = Metric("mean")
    metric_sum = Metric("sum")
    # Compute gt metrics
    gt_mean = metric_mean.compute_metric(classif_map)
    gt_sum = metric_sum.compute_metric(classif_map)
    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(
        classification_layer=["Slope0"], metrics=["mean", "sum"]
    )
    # Get output metrics
    output_mean = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="exclusion",
        metric="mean",
    )
    output_sum = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="exclusion",
        metric="sum",
    )
    # Test that the output metrics are the same as gt
    np.testing.assert_allclose(
        gt_mean,
        output_mean,
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_sum,
        output_sum,
        rtol=1e-02,
    )
    # Do the same test with the ratio_above_threshold 2D metric
    # Initialize metric objects
    elevation_thrlds = [-3, 2, 90]
    metric_ratio = Metric(
        "ratio_above_threshold",
        params={"elevation_threshold": elevation_thrlds},
    )
    # Compute gt metrics
    gt_ratio = metric_ratio.compute_metric(classif_map)
    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(
        classification_layer=["Slope0"],
        metrics=[
            {"ratio_above_threshold": {"elevation_threshold": elevation_thrlds}}
        ],
    )
    # Get output metrics
    output_ratio = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="exclusion",
        metric="ratio_above_threshold",
    )
    # Test that the output metrics are the same as gt
    np.testing.assert_allclose(
        gt_ratio,
        output_ratio,
        rtol=1e-1,
    )


@pytest.mark.unit_tests
@pytest.mark.functional_tests
def test_compute_stats_from_cfg_slope(initialize_stats_processing_with_metrics):
    """
    Tests that the initialization of StatsProcessing
    with an input configuration containing metrics correctly
    computes the specified metrics for
    the slope classification layer
    Input data:
    - StatsProcessing object from the
      "initialize_stats_processing_with_metrics" fixture
    Validation data:
    - The manually computed statistics on the pixels
      corresponding to the class 1 of the slope classification
      layer: gt_std, gt_nmad
    Validation process:
    - Creates the StatsProcessing object
    - Computes the nodata_nan_mask of the StatsProcessing's
      input dem
    - Computes the ground truth metrics on the valid pixels
      corresponding to the class 1
    - Computes the metrics using the StatsProcessing's compute_stats
      function and gets the result using the get_classification_layer_metric
      function
    - Checks that the obtained metrics are the same as ground truth
    - Checks that the ratio metric has not been computed
    - Checked function: StatsProcessing's initialization,
      compute_stats and get_classification_layer_metric
    """
    # Initialize stats processing object with metrics on the cfg
    stats_processing = initialize_stats_processing_with_metrics

    # Compute stats for all classif layers and its metrics
    stats_dataset = stats_processing.compute_stats()
    # ----------------------------------------------------------
    # TEST Slope layer class 1
    classif_class = 1
    dataset_idx = stats_processing.classification_layers_names.index("Slope0")
    # Manually compute mean and sum stats

    # Initialize metric objects
    metric_std = Metric("std")
    metric_nmad = Metric("nmad")

    # Compute nodata_nan_mask
    nodata_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != nodata_value),
        0,
        stats_processing.dem["image"].data,
    )
    # Get valid class indexes for status dataset and class 1
    idxes_map_class_1_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[
            "ref"
        ][classif_class]
    )
    # Get class pixels
    classif_map = stats_processing.dem["image"].data[
        idxes_map_class_1_dataset_0
    ]
    # Compute gt metrics
    gt_std = metric_std.compute_metric(classif_map)
    gt_nmad = metric_nmad.compute_metric(classif_map)

    # Get output metrics
    output_std = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="standard",
        metric="std",
    )
    output_nmad = stats_dataset.get_classification_layer_metric(
        classification_layer="Slope0",
        classif_class=classif_class,
        mode="standard",
        metric="nmad",
    )
    # Test that the output metrics are the same as gt
    np.testing.assert_allclose(
        gt_std,
        output_std,
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_nmad,
        output_nmad,
        rtol=1e-02,
    )

    # Test that the ratio metric has not been computed for the Slope0 dataset
    # Test that an error is raised
    with pytest.raises(KeyError):
        stats_dataset.get_classification_layer_metric(
            classification_layer="Slope0",
            classif_class=classif_class,
            mode="standard",
            metric="ratio_above_threshold",
        )
