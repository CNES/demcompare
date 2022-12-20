#!/usr/bin/env python
# coding: utf8
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
get_classification_metric(s) functions.
"""
# pylint:disable = duplicate-code
# Standard imports

import os

# Third party imports
import pytest

import demcompare
from demcompare import dem_tools
from demcompare.helpers_init import read_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path


@pytest.mark.unit_tests
@pytest.mark.functional_tests
def test_get_classification_layer_metric(initialize_stats_dataset):
    """
    Test the get_classification_layer_metric function.
    Input data:
    - StatsDataset from the "initialize_stats_dataset" fixture
    Validation data:
    - Manually obtained metric value by accessing the corresponding
      StatsDataset attributes and indexes: gt_mean_standard_class_0,
      output_sum_intersection_class_1, output_mean_standard_all_class,
      output_sum_intersection_all_class
    Validation process:
    - Check that the metric value obtained with the
      "get_classification_layer_metric" is the same as ground truth
    - Checked function : StatsDataset's
      get_classification_layer_metric
    """
    # Initialize input stats dict for two modes
    stats_dataset = initialize_stats_dataset

    # Test specifying the class ------------------------
    # Get the mean metric for class 0 and mode standard
    dataset_idx = 0

    output_mean_standard_class_0 = (
        stats_dataset.get_classification_layer_metric(
            classification_layer="Status",
            classif_class=0,
            mode="standard",
            metric="mean",
        )
    )
    gt_mean_standard_class_0 = stats_dataset.classif_layers_dataset[
        dataset_idx
    ].attrs["stats_by_class"][0]["mean"]
    assert gt_mean_standard_class_0 == output_mean_standard_class_0

    # Get the mean metric for class 1 and mode intersection

    output_sum_intersection_class_1 = (
        stats_dataset.get_classification_layer_metric(
            classification_layer="Status",
            classif_class=1,
            mode="intersection",
            metric="sum",
        )
    )
    gt_sum_intersection_class_1 = stats_dataset.classif_layers_dataset[
        dataset_idx
    ].attrs["stats_by_class_intersection"][1]["sum"]
    assert gt_sum_intersection_class_1 == output_sum_intersection_class_1

    # Test without specifying the class
    # (it returns a list for all classes) ------------------------

    # Get the mean metric for all classes and mode standard

    output_mean_standard_all_class = (
        stats_dataset.get_classification_layer_metric(
            classification_layer="Status", mode="standard", metric="mean"
        )
    )
    gt_mean_standard_all_class = [
        stats_dataset.classif_layers_dataset[dataset_idx].attrs[
            "stats_by_class"
        ][0]["mean"],
        stats_dataset.classif_layers_dataset[dataset_idx].attrs[
            "stats_by_class"
        ][1]["mean"],
    ]
    assert gt_mean_standard_all_class == output_mean_standard_all_class

    # Get the mean metric for all classes and mode intersection

    output_sum_intersection_all_class = (
        stats_dataset.get_classification_layer_metric(
            classification_layer="Status", mode="intersection", metric="sum"
        )
    )
    gt_sum_intersection_all_class = [
        stats_dataset.classif_layers_dataset[dataset_idx].attrs[
            "stats_by_class_intersection"
        ][0]["sum"],
        stats_dataset.classif_layers_dataset[dataset_idx].attrs[
            "stats_by_class_intersection"
        ][1]["sum"],
    ]
    assert gt_sum_intersection_all_class == output_sum_intersection_all_class


@pytest.mark.unit_tests
@pytest.mark.functional_tests
def test_get_classification_layer_metrics(initialize_stats_dataset):
    """
    Test the get_classification_layer_metrics function.
    Input data:
    - StatsDataset from the "initialize_stats_dataset" fixture
    Validation data:
    - Names of all metrics that are available in the StatsDataset:
      gt_available_metrics
    Validation process:
    - Check that the metric names obtained with the
      "get_classification_layer_metrics" are the same as ground truth
    - Checked function : StatsDataset's
      get_classification_layer_metrics
    """
    # Initialize stats_dataset
    stats_dataset = initialize_stats_dataset

    # Test specifying the class ------------------------
    # Get the mean metric for class 0 and mode standard

    output_available_metrics = stats_dataset.get_classification_layer_metrics(
        classification_layer="Status"
    )
    gt_available_metrics = [
        "mean",
        "sum",
        "nbpts",
        "percent_valid_points",
    ]
    assert gt_available_metrics == output_available_metrics


@pytest.mark.unit_tests
def test_get_classification_layer_metrics_from_stats_processing():
    """
    Tests the get_classification_layer_metrics function.
    Manually computes input stats via the StatsProcessing.compute_stats
    API and tests that the get_classification_layer_metrics function
    correctly returns the metric names.
    Input data:
    - Manually created StatsProcessing with two classification layers,
      one global and one segmentation named "Status"
    Validation data:
    - Names of all metrics that are available in the StatsDataset
      for each classification layer: gt_metrics_global, gt_metrics_status
    Validation process:
    - Obtain StatsDataset from the StatsProcessing.compute_stats
    - Compute new metrics using the API of the StatsProcessing object
    - Check that the metric names obtained with the
      "get_classification_layer_metrics" are the same as ground truth
      (hence the new metrics computed via the API have been added to the
      StatsDataset)
    - Checked function : StatsDataset's
      get_classification_layer_metrics
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Initialize sec and ref, necessary for StatsProcessing creation
    sec = dem_tools.load_dem(cfg["input_sec"]["path"])
    ref = dem_tools.load_dem(
        cfg["input_ref"]["path"],
        classification_layers=(cfg["input_ref"]["classification_layers"]),
    )
    sec, ref, _ = dem_tools.reproject_dems(sec, ref, sampling_source="ref")

    # Compute altitude diff for stats computation
    stats_dem = dem_tools.compute_alti_diff_for_stats(ref, sec)
    # Initialize stats input configuration
    input_stats_cfg = {
        "remove_outliers": "False",
        "classification_layers": {
            "Status": {
                "type": "segmentation",
                "classes": {
                    "valid": [0],
                    "KO": [1],
                    "Land": [2],
                    "NoData": [3],
                    "Outside_detector": [4],
                },
            }
        },
    }
    # Create StatsProcessing object
    stats_processing = demcompare.StatsProcessing(input_stats_cfg, stats_dem)

    # Compute stats from input cfg  -----------------------------------------

    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(metrics=["mean", "sum"])
    # Define gt metric names
    gt_metrics_global = ["mean", "sum", "nbpts", "percent_valid_points"]
    gt_metrics_status = ["mean", "sum", "nbpts", "percent_valid_points"]
    # Compute output metric names
    output_metrics_global = stats_dataset.get_classification_layer_metrics(
        "global"
    )
    output_metrics_status = stats_dataset.get_classification_layer_metrics(
        "Status"
    )
    # Verify that the metrics are correctly obtained
    assert output_metrics_global == gt_metrics_global
    assert output_metrics_status == gt_metrics_status

    # Compute an additional metric on one of the datasets
    elevation_thrlds = [-3, 2, 90]
    stats_dataset = stats_processing.compute_stats(
        classification_layer=["Status"],
        metrics=[
            {"ratio_above_threshold": {"elevation_threshold": elevation_thrlds}}
        ],
    )
    # Define gt metric names
    gt_metrics_global = ["mean", "sum", "nbpts", "percent_valid_points"]
    gt_metrics_status = [
        "mean",
        "sum",
        "nbpts",
        "percent_valid_points",
        "ratio_above_threshold",
    ]
    # Compute output metric names
    output_metrics_global = stats_dataset.get_classification_layer_metrics(
        "global"
    )
    output_metrics_status = stats_dataset.get_classification_layer_metrics(
        "Status"
    )

    # Verify that the status metrics have been updated
    assert output_metrics_global == gt_metrics_global
    assert output_metrics_status == gt_metrics_status
