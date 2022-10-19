#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
methods in the StatsPair class.
"""

# Standard imports

import os

import numpy as np

# Third party imports
import pytest

import demcompare
from demcompare import dem_tools
from demcompare.helpers_init import read_config_file
from demcompare.stats_dataset import StatsDataset

# Tests helpers
from .helpers import demcompare_test_data_path


# Tests helpers
@pytest.fixture(name="input_images")
def fixture_input_images():
    """
    Fixture to initialize the input images used for the
    test stats_dataset
    """
    # Create input arrays
    sec = np.array(
        [[1, 1, 1], [1, 1, -33]],
        dtype=np.float32,
    )
    ref = np.array(
        [
            [3, 3, 3],
            [1, 1, 0],
        ],
        dtype=np.float32,
    )
    return ref, sec


@pytest.fixture(name="input_stats_status_results")
def fixture_input_stats_status_results():
    """
    Fixture to initialize the Status layer statistics used for the
    test stats_dataset
    """
    input_stats_status_standard = [
        {
            "mean": 26,
            "sum": 5,
            "nbpts": 2,
            "dz_values": np.array([[10, 10, 10], [0, 0, 1]], dtype=np.float32),
            "class_name": "valid:[0]",
            "percent_valid_points": 0.22,
        },
        {
            "mean": -9.5,
            "sum": -28,
            "nbpts": 29,
            "dz_values": np.array([[10, 7, 10], [0, 0, 2]], dtype=np.float32),
            "class_name": "KO:[1]",
            "percent_valid_points": 3.13,
        },
    ]
    input_stats_status_intersection = [
        {
            "mean": 26.21,
            "sum": 556,
            "nbpts": 2122,
            "dz_values": np.array([[10, 10, 10], [0, 0, 3]], dtype=np.float32),
            "class_name": "valid:[0]",
            "percent_valid_points": 0.22,
        },
        {
            "mean": -9.57,
            "sum": -280203.68,
            "nbpts": 29,
            "dz_values": np.array([[10, 10, 10], [0, 0, 4]], dtype=np.float32),
            "class_name": "KO:[1]",
            "percent_valid_points": 3.1,
        },
    ]
    return input_stats_status_standard, input_stats_status_intersection


@pytest.fixture(name="input_stats_slope_results")
def fixture_input_stats_slope_results():
    """
    Fixture to initialize the Slope0 layer statistics used for the
    test stats_dataset
    """
    # Initialize input stats dict for Slope class and three modes
    input_stats_slope_standard = [
        {
            "mean": 3,
            "sum": 4,
            "nbpts": 5,
            "dz_values": np.array([[-10, 10, 10], [0, 0, 4]], dtype=np.float32),
            "class_name": "[0%;10%[:0",
            "percent_valid_points": 0.66,
        },
        {
            "mean": -9.5,
            "sum": -28,
            "nbpts": 29,
            "dz_values": np.array([[-10, 10, 10], [0, 0, 4]], dtype=np.float32),
            "class_name": "KO:[1]",
            "percent_valid_points": 3.13,
        },
    ]
    input_stats_slope_intersection = [
        {
            "mean": 5,
            "sum": 8,
            "nbpts": 3,
            "dz_values": np.array([[-10, 10, 10], [0, 0, 4]], dtype=np.float32),
            "class_name": "[10%;25%[:10",
            "percent_valid_points": 0.4,
        },
        {
            "mean": -9,
            "sum": -2802,
            "nbpts": 9,
            "dz_values": np.array([[-10, 10, 10], [0, 0, 4]], dtype=np.float32),
            "class_name": "KO:[1]",
            "percent_valid_points": 3.1,
        },
    ]
    input_stats_slope_exclusion = [
        {
            "mean": 2,
            "sum": 6,
            "nbpts": 21,
            "dz_values": np.array([[-10, 10, 10], [0, 0, 5]], dtype=np.float32),
            "class_name": "[10%;25%[:10",
            "percent_valid_points": 0.2,
        },
        {
            "mean": -9.77,
            "sum": -2,
            "nbpts": 289,
            "dz_values": np.array([[-10, 10, 10], [0, 0, 6]], dtype=np.float32),
            "class_name": "KO:[1]",
            "percent_valid_points": 3,
        },
    ]
    return (
        input_stats_slope_standard,
        input_stats_slope_intersection,
        input_stats_slope_exclusion,
    )


@pytest.fixture(name="initialize_stats_dataset")
def fixture_initialize_stats_dataset(
    input_stats_status_results, input_stats_slope_results, input_images
):
    """
    Fixture to initialize the StatsDataset object for tests
    """
    (
        input_stats_status_standard,
        input_stats_status_intersection,
    ) = input_stats_status_results

    # Initialize StatsDataset
    ref, sec = input_images
    stats_dataset = StatsDataset(sec - ref)

    # Add stats for each mode and class Status
    stats_dataset.add_classif_layer_and_mode_stats(
        classif_name="Status",
        input_stats=input_stats_status_standard,
        mode_name="standard",
    )
    stats_dataset.add_classif_layer_and_mode_stats(
        classif_name="Status",
        input_stats=input_stats_status_intersection,
        mode_name="intersection",
    )

    (
        input_stats_slope_standard,
        input_stats_slope_intersection,
        input_stats_slope_exclusion,
    ) = input_stats_slope_results

    # Add stats for each mode and class Slope
    stats_dataset.add_classif_layer_and_mode_stats(
        classif_name="Slope0",
        input_stats=input_stats_slope_standard,
        mode_name="standard",
    )
    stats_dataset.add_classif_layer_and_mode_stats(
        classif_name="Slope0",
        input_stats=input_stats_slope_intersection,
        mode_name="intersection",
    )
    stats_dataset.add_classif_layer_and_mode_stats(
        classif_name="Slope0",
        input_stats=input_stats_slope_exclusion,
        mode_name="exclusion",
    )

    return stats_dataset


@pytest.mark.unit_tests
def test_add_classif_layer_and_mode_stats_names(initialize_stats_dataset):
    """
    Test the add_classif_layer_and_mode_stats function.
    Manually computes input stats for two classification
    layers and different modes, and tests that the
    add_classif_layer_and_mode_stats function correctly
    adds the dataset names.
    """
    stats_dataset = initialize_stats_dataset

    # Test dataset_names attribute
    gt_dataset_names = ["Status", "Slope0"]
    assert (
        list(stats_dataset.classif_layers_and_modes.keys()) == gt_dataset_names
    )


@pytest.mark.unit_tests
def test_add_classif_layer_and_mode_stats_status_layer(
    initialize_stats_dataset, input_stats_status_results, input_images
):
    """
    Test the add_classif_layer_and_mode_stats function.
    Manually computes input stats for two classification
    layers and different modes, and tests that the
    add_classif_layer_and_mode_stats function correctly
    adds the Status layer information on the stats_dataset.

    Also indirectly tests the get_dataset function.
    """
    stats_dataset = initialize_stats_dataset
    (
        input_stats_status_standard,
        input_stats_status_intersection,
    ) = input_stats_status_results
    # Get input arrays
    ref, sec = input_images

    # Define alti diff
    gt_image = sec - ref
    # Status dataset  --------------------------------------------------------
    # Get the status_dataset xarray dataset
    status_dataset = stats_dataset.get_classification_layer_dataset("Status")
    # Test that the metrics of the class 0 have been correctly set
    class_idx = 0

    # Test mean and nbpts global metrics
    # Standard mode
    assert (
        status_dataset.attrs["stats_by_class"][class_idx]["mean"]
        == input_stats_status_standard[class_idx]["mean"]
    )
    assert (
        status_dataset.attrs["stats_by_class"][class_idx]["nbpts"]
        == input_stats_status_standard[class_idx]["nbpts"]
    )
    # Intersection mode
    assert (
        status_dataset.attrs["stats_by_class_intersection"][class_idx]["mean"]
        == input_stats_status_intersection[class_idx]["mean"]
    )
    assert (
        status_dataset.attrs["stats_by_class_intersection"][class_idx]["nbpts"]
        == input_stats_status_intersection[class_idx]["nbpts"]
    )

    # Test alti diff
    np.testing.assert_allclose(
        status_dataset.image.data,
        gt_image,
        rtol=1e-02,
    )
    # Test alti diff by class
    # Standard mode
    np.testing.assert_allclose(
        status_dataset.image_by_class.data[:, :, class_idx],
        input_stats_status_standard[class_idx]["dz_values"],
        rtol=1e-02,
    )
    # Intersection mode
    np.testing.assert_allclose(
        status_dataset.image_by_class_intersection.data[:, :, class_idx],
        input_stats_status_intersection[class_idx]["dz_values"],
        rtol=1e-02,
    )

    class_idx = 1  # ----------------------------
    # Test that the metrics of the class 1 have been correctly set

    # Test mean and nbpts global metrics
    # Standard mode
    assert (
        status_dataset.attrs["stats_by_class"][class_idx]["mean"]
        == input_stats_status_standard[class_idx]["mean"]
    )
    assert (
        status_dataset.attrs["stats_by_class"][class_idx]["nbpts"]
        == input_stats_status_standard[class_idx]["nbpts"]
    )
    # Intersection mode
    assert (
        status_dataset.attrs["stats_by_class_intersection"][class_idx]["mean"]
        == input_stats_status_intersection[class_idx]["mean"]
    )
    assert (
        status_dataset.attrs["stats_by_class_intersection"][class_idx]["nbpts"]
        == input_stats_status_intersection[class_idx]["nbpts"]
    )

    # Test alti diff by class
    # Standard mode
    np.testing.assert_allclose(
        status_dataset.image_by_class.data[:, :, class_idx],
        input_stats_status_standard[class_idx]["dz_values"],
        rtol=1e-02,
    )
    # Intersection mode
    np.testing.assert_allclose(
        status_dataset.image_by_class_intersection[:, :, class_idx],
        input_stats_status_intersection[class_idx]["dz_values"],
        rtol=1e-02,
    )


@pytest.mark.unit_tests
def test_add_classif_layer_and_mode_stats_slope_layer(
    initialize_stats_dataset, input_stats_slope_results, input_images
):
    """
    Test the add_classif_layer_and_mode_stats function.
    Manually computes input stats for two classification
    layers and different modes, and tests that the
    add_classif_layer_and_mode_stats function correctly
    adds the Slope layer information on the stats_dataset.

    Also indirectly tests the get_dataset function.
    """
    stats_dataset = initialize_stats_dataset
    (
        input_stats_slope_standard,
        input_stats_slope_intersection,
        input_stats_slope_exclusion,
    ) = input_stats_slope_results
    # Get input arrays
    ref, sec = input_images

    # Define alti diff
    gt_image = sec - ref
    # Slope dataset  --------------------------------------------------------
    # Get the slope_dataset xarray dataset

    slope_dataset = stats_dataset.get_classification_layer_dataset("Slope0")

    class_idx = 0  # ----------------------------
    # Test that the metrics of the class 0 have been correctly set

    # Test mean and nbpts global metrics
    # Standard mode
    assert (
        slope_dataset.attrs["stats_by_class"][class_idx]["mean"]
        == input_stats_slope_standard[class_idx]["mean"]
    )
    assert (
        slope_dataset.attrs["stats_by_class"][class_idx]["nbpts"]
        == input_stats_slope_standard[class_idx]["nbpts"]
    )
    # Intersection mode
    assert (
        slope_dataset.attrs["stats_by_class_intersection"][class_idx]["mean"]
        == input_stats_slope_intersection[class_idx]["mean"]
    )
    assert (
        slope_dataset.attrs["stats_by_class_intersection"][class_idx]["nbpts"]
        == input_stats_slope_intersection[class_idx]["nbpts"]
    )
    # Exclusion mode
    assert (
        slope_dataset.attrs["stats_by_class_exclusion"][class_idx]["mean"]
        == input_stats_slope_exclusion[class_idx]["mean"]
    )
    assert (
        slope_dataset.attrs["stats_by_class_exclusion"][class_idx]["nbpts"]
        == input_stats_slope_exclusion[class_idx]["nbpts"]
    )
    # Test alti diff
    np.testing.assert_allclose(
        slope_dataset.image.data,
        gt_image,
        rtol=1e-02,
    )
    # Test alti diff by class
    # Standard mode
    np.testing.assert_allclose(
        slope_dataset.image_by_class.data[:, :, class_idx],
        input_stats_slope_standard[class_idx]["dz_values"],
        rtol=1e-02,
    )
    # Intersection mode
    np.testing.assert_allclose(
        slope_dataset.image_by_class_intersection.data[:, :, class_idx],
        input_stats_slope_intersection[class_idx]["dz_values"],
        rtol=1e-02,
    )
    # Exclusion mode
    np.testing.assert_allclose(
        slope_dataset.image_by_class_exclusion.data[:, :, class_idx],
        input_stats_slope_exclusion[class_idx]["dz_values"],
        rtol=1e-02,
    )

    class_idx = 1  # ----------------------------
    # Test that the metrics of the class 1 have been correctly set

    # Test mean and nbpts global metrics
    # Standard mode
    assert (
        slope_dataset.attrs["stats_by_class"][class_idx]["mean"]
        == input_stats_slope_standard[class_idx]["mean"]
    )
    assert (
        slope_dataset.attrs["stats_by_class"][class_idx]["nbpts"]
        == input_stats_slope_standard[class_idx]["nbpts"]
    )
    # Intersection mode
    assert (
        slope_dataset.attrs["stats_by_class_intersection"][class_idx]["mean"]
        == input_stats_slope_intersection[class_idx]["mean"]
    )
    assert (
        slope_dataset.attrs["stats_by_class_intersection"][class_idx]["nbpts"]
        == input_stats_slope_intersection[class_idx]["nbpts"]
    )
    # Exclusion mode
    assert (
        slope_dataset.attrs["stats_by_class_exclusion"][class_idx]["mean"]
        == input_stats_slope_exclusion[class_idx]["mean"]
    )
    assert (
        slope_dataset.attrs["stats_by_class_exclusion"][class_idx]["nbpts"]
        == input_stats_slope_exclusion[class_idx]["nbpts"]
    )

    # Test alti diff by class
    # Standard mode
    np.testing.assert_allclose(
        slope_dataset.image_by_class.data[:, :, class_idx],
        input_stats_slope_standard[class_idx]["dz_values"],
        rtol=1e-02,
    )
    # Intersection mode
    np.testing.assert_allclose(
        slope_dataset.image_by_class_intersection.data[:, :, class_idx],
        input_stats_slope_intersection[class_idx]["dz_values"],
        rtol=1e-02,
    )
    # Exclusion mode
    np.testing.assert_allclose(
        slope_dataset.image_by_class_exclusion.data[:, :, class_idx],
        input_stats_slope_exclusion[class_idx]["dz_values"],
        rtol=1e-02,
    )


@pytest.mark.unit_tests
def test_get_classification_layer_metric(initialize_stats_dataset):
    """
    Test the get_classification_layer_metric function.
    Manually computes input stats for one classification
    layers and different modes, and tests that the
    get_classification_layer_metric function correctly
    returns the corresponding metric.
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

    output_mean_standard_class_0 = (
        stats_dataset.get_classification_layer_metric(
            classification_layer="Status", mode="standard", metric="mean"
        )
    )
    gt_mean_standard_class_0 = [
        stats_dataset.classif_layers_dataset[dataset_idx].attrs[
            "stats_by_class"
        ][0]["mean"],
        stats_dataset.classif_layers_dataset[dataset_idx].attrs[
            "stats_by_class"
        ][1]["mean"],
    ]
    assert gt_mean_standard_class_0 == output_mean_standard_class_0

    # Get the mean metric for all classes and mode intersection

    output_sum_intersection_class_1 = (
        stats_dataset.get_classification_layer_metric(
            classification_layer="Status", mode="intersection", metric="sum"
        )
    )
    gt_sum_intersection_class_1 = [
        stats_dataset.classif_layers_dataset[dataset_idx].attrs[
            "stats_by_class_intersection"
        ][0]["sum"],
        stats_dataset.classif_layers_dataset[dataset_idx].attrs[
            "stats_by_class_intersection"
        ][1]["sum"],
    ]
    assert gt_sum_intersection_class_1 == output_sum_intersection_class_1


@pytest.mark.unit_tests
def test_get_classification_layer_metrics(initialize_stats_dataset):
    """
    Test the get_classification_layer_metrics function.
    Manually computes input stats for one classification
    layers and different modes, and tests that the
    get_classification_layer_metrics function correctly
    returns the corresponding metric names.
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
    Manually computes input stats for one classification
    layer and different modes, then computes more stats via the
    StatsProcessing.compute_stats API and tests that the
    get_classification_layer_metrics function correctly
    returns the metric names.
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
