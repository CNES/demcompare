#!/usr/bin/env python
# coding: utf8
# Disable the protected-access to test the functions
# pylint:disable=protected-access
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
methods in the StatsProcessing class.
"""
# pylint:disable = duplicate-code
# pylint:disable = too-many-lines
# Standard imports
import os

# Third party imports
import numpy as np
import pytest

# Demcompare imports
import demcompare
from demcompare import dem_tools
from demcompare.classification_layer import (
    ClassificationLayer,
    FusionClassificationLayer,
)
from demcompare.initialization import read_config_file
from demcompare.metric import Metric

# Tests helpers
from .helpers import demcompare_test_data_path


@pytest.mark.unit_tests
def test_create_classif_layers():
    """
    Test the create_classif_layers function
    Creates a StatsProcessing object with an input configuration
    and tests that its created classification layers
    are the same as gt.

    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")

    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Initialize sec and ref
    sec = dem_tools.load_dem(
        cfg["input_sec"]["path"],
        classification_layers=(cfg["input_sec"]["classification_layers"]),
    )
    ref = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec, ref, _ = dem_tools.reproject_dems(sec, ref)

    # Compute slope and add it as a classification_layer
    ref = dem_tools.compute_dem_slope(ref)
    sec = dem_tools.compute_dem_slope(sec)
    # Compute altitude diff for stats computation
    stats_dem = dem_tools.compute_alti_diff_for_stats(ref, sec)

    # Initialize stats input configuration
    input_stats_cfg = {
        "remove_outliers": "False",
        "save_results": "False",
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
            },
            "Slope0": {
                "type": "slope",
                "ranges": [0, 10, 25, 50, 90],
            },
            "fusion": {"sec": ["Slope0", "Status"]},
        },
    }
    # Create StatsProcessing object
    stats_processing = demcompare.StatsProcessing(input_stats_cfg, stats_dem)

    # Create ground truth Status layer
    gt_status_layer = ClassificationLayer(
        "Status",
        "segmentation",
        stats_dem,
        {
            "type": "segmentation",
            "classes": {
                "valid": [0],
                "KO": [1],
                "Land": [2],
                "NoData": [3],
                "Outside_detector": [4],
            },
        },
    )
    # Create ground truth Slope layer
    gt_slope_layer = ClassificationLayer(
        "Slope0",
        "slope",
        stats_dem,
        {"type": "slope", "ranges": [0, 10, 25, 50, 90]},
    )
    # Create ground truth Global layer
    gt_global_layer = ClassificationLayer(
        "global",
        "global",
        stats_dem,
        {"type": "global"},
    )
    # Create ground truth Fusion layer
    gt_fusion_layer = FusionClassificationLayer(
        [gt_status_layer, gt_slope_layer], map_idx=1
    )
    # Get StatsProcessing created classification layers
    output_classification_layers = stats_processing.classification_layers

    # Test that StatsProcessing's classification layer's cfg is the same
    # as the gt layer's cfg
    assert output_classification_layers[0].cfg == gt_status_layer.cfg
    assert output_classification_layers[1].cfg == gt_slope_layer.cfg
    assert output_classification_layers[2].cfg == gt_global_layer.cfg
    assert output_classification_layers[3].cfg == gt_fusion_layer.cfg


@pytest.mark.unit_tests
def test_create_classif_layers_without_input_classif():
    """
    Test the create_classif_layers function
    Creates a StatsProcessing object with an input configuration
    that does not specify any classification layer
    and tests that its created classification layers
    are the same as the default gt.

    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")

    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Initialize sec and ref, necessary for StatsProcessing creation
    sec = dem_tools.load_dem(
        cfg["input_sec"]["path"],
        classification_layers=(cfg["input_sec"]["classification_layers"]),
    )
    ref = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec, ref, _ = dem_tools.reproject_dems(sec, ref)

    # Compute slope and add it as a classification_layer
    ref = dem_tools.compute_dem_slope(ref)
    sec = dem_tools.compute_dem_slope(sec)
    # Compute altitude diff for stats computation
    # Compute altitude diff for stats computation
    stats_dem = dem_tools.compute_alti_diff_for_stats(ref, sec)

    # Initialize stats input configuration
    input_stats_cfg = {
        "remove_outliers": "False",
        "save_results": "False",
    }

    # Create StatsProcessing object
    stats_processing = demcompare.StatsProcessing(input_stats_cfg, stats_dem)

    # Create ground truth default Global layer
    gt_global_layer = ClassificationLayer(
        "global",
        "global",
        stats_dem,
        {"type": "global"},
    )

    # Get StatsProcessing created classification layers
    output_classification_layers = stats_processing.classification_layers
    # Test that StatsProcessing's classification layer's cfg is the same
    # as the gt layer's cfg
    assert output_classification_layers[0].cfg == gt_global_layer.cfg


@pytest.mark.unit_tests
def test_compute_stats():
    """
    Tests the compute_stats. Manually computes
    the stats for the classification_layers for a given class
    and mode and tests that the compute_stats function obtains
    the same values.
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

    # Compute slope and add it as a classification_layer
    ref = dem_tools.compute_dem_slope(ref)
    sec = dem_tools.compute_dem_slope(sec)
    # Compute altitude diff for stats computation
    stats_dem = dem_tools.compute_alti_diff_for_stats(ref, sec)
    # Initialize stats input configuration
    input_stats_cfg = {
        "remove_outliers": "False",
        "save_results": "False",
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
            },
            "Slope0": {
                "type": "slope",
                "ranges": [0, 10, 25, 50, 90],
            },
            "fusion": {"ref": ["Slope0", "Status"]},
        },
    }
    # Create StatsProcessing object
    stats_processing = demcompare.StatsProcessing(input_stats_cfg, stats_dem)

    # ----------------------------------------------------------
    # TEST status layer class 1
    classif_class = 1
    dataset_idx = stats_processing.classification_layers_names.index("Status")
    # Manually compute mean and sum stats

    # Initialize metric objects
    metric_mean = Metric("mean")
    metric_sum = Metric("sum")

    # Compute nodata_nan_mask
    no_data_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != no_data_value),
        0,
        stats_processing.dem["image"].data,
    )
    # Get valid class indexes for status dataset and class 1
    idxes_map_class_1_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[0][
            classif_class
        ]
    )
    # Get class pixels
    classif_map = stats_processing.dem["image"].data[
        idxes_map_class_1_dataset_0
    ]
    # Compute gt metrics
    gt_mean = metric_mean.compute_metric(classif_map)
    gt_sum = metric_sum.compute_metric(classif_map)
    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(metrics=["mean", "sum"])
    # Get output metrics
    output_mean = stats_dataset.get_classification_layer_metric(
        classification_layer="Status",
        classif_class=classif_class,
        mode="standard",
        metric="mean",
    )
    output_sum = stats_dataset.get_classification_layer_metric(
        classification_layer="Status",
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
        classification_layer=["Status"],
        metrics=[
            {"ratio_above_threshold": {"elevation_threshold": elevation_thrlds}}
        ],
    )
    # Get output metrics
    output_ratio = stats_dataset.get_classification_layer_metric(
        classification_layer="Status",
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
    # ----------------------------------------------------------
    # Test global layer class 0
    classif_class = 0
    dataset_idx = stats_processing.classification_layers_names.index("global")

    # Manually compute mean and sum stats
    # Get alti map
    classif_map = stats_processing.dem["image"].data

    # Compute nodata_nan_mask
    no_data_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != no_data_value),
        0,
        stats_processing.dem["image"].data,
    )
    # Class pixels of the global layer are all valid pixels
    # Compute gt metrics
    gt_mean = metric_mean.compute_metric(classif_map[np.where(nodata_nan_mask)])
    gt_sum = metric_sum.compute_metric(classif_map[np.where(nodata_nan_mask)])

    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(metrics=["mean", "sum"])
    # Get output metrics
    output_mean = stats_dataset.get_classification_layer_metric(
        classification_layer="global",
        classif_class=classif_class,
        mode="standard",
        metric="mean",
    )
    output_sum = stats_dataset.get_classification_layer_metric(
        classification_layer="global",
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
    # Compute gt metrics
    gt_ratio = metric_ratio.compute_metric(
        classif_map[np.where(nodata_nan_mask)]
    )
    # Compute stats with stats processing
    stats_dataset = stats_processing.compute_stats(
        classification_layer=["global"],
        metrics=[
            {"ratio_above_threshold": {"elevation_threshold": elevation_thrlds}}
        ],
    )
    # Test that the output metrics are the same as gt
    output_ratio = stats_dataset.get_classification_layer_metric(
        classification_layer="global",
        classif_class=classif_class,
        mode="standard",
        metric="ratio_above_threshold",
    )

    np.testing.assert_allclose(
        gt_ratio,
        output_ratio,
        rtol=1e-02,
    )
    # Test slope layer class 0
    # ----------------------------------------------------------
    classif_class = 0
    dataset_idx = stats_processing.classification_layers_names.index("Slope0")

    # Compute nodata_nan_mask
    no_data_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != no_data_value),
        0,
        stats_processing.dem["image"].data,
    )
    # Get valid class indexes for slope dataset and class 1
    idxes_map_class_0_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[0][
            classif_class
        ]
    )
    # Get class pixels
    classif_map = stats_processing.dem["image"].data[
        idxes_map_class_0_dataset_0
    ]

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
def test_compute_stats_slope_classif_intersection_mode():
    """
    Tests the compute_stats. Manually computes
    the stats for the slope function for a given class
    and intersection mode and tests that the
    compute_stats function obtains
    the same values.
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
    sec, ref, _ = dem_tools.reproject_dems(sec, ref)

    # Compute slope and add it as a classification_layer
    ref = dem_tools.compute_dem_slope(ref)
    sec = dem_tools.compute_dem_slope(sec)
    # Compute altitude diff for stats computation
    stats_dem = dem_tools.compute_alti_diff_for_stats(ref, sec)
    # Initialize stats input configuration
    input_stats_cfg = {
        "remove_outliers": "False",
        "save_results": "False",
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
            },
            "Slope0": {
                "type": "slope",
                "ranges": [0, 10, 25, 50, 90],
            },
        },
    }
    # Create StatsProcessing object
    stats_processing = demcompare.StatsProcessing(input_stats_cfg, stats_dem)

    # Test slope layer class 0
    # ----------------------------------------------------------
    classif_class = 0
    dataset_idx = stats_processing.classification_layers_names.index("Slope0")

    # Compute nodata_nan_mask
    no_data_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != no_data_value),
        0,
        stats_processing.dem["image"].data,
    )

    # Get valid class indexes for slope dataset and class 1 mode intersection
    idxes_map_class_0_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[0][
            classif_class
        ]
        * stats_processing.classification_layers[dataset_idx].classes_masks[1][
            classif_class
        ]
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
def test_compute_stats_slope_classif_exclusion_mode():
    """
    Tests the compute_stats. Manually computes
    the stats for the slope function for a given class
    and exclusion mode and tests that the
    compute_stats function obtains
    the same values.
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Initialize sec and ref
    sec = dem_tools.load_dem(cfg["input_sec"]["path"])
    ref = dem_tools.load_dem(
        cfg["input_ref"]["path"],
        classification_layers=(cfg["input_ref"]["classification_layers"]),
    )
    sec, ref, _ = dem_tools.reproject_dems(sec, ref)

    # Compute slope and add it as a classification_layer
    ref = dem_tools.compute_dem_slope(ref)
    sec = dem_tools.compute_dem_slope(sec)
    # Compute altitude diff for stats computation
    stats_dem = dem_tools.compute_alti_diff_for_stats(ref, sec)
    # Initialize stats input configuration
    input_stats_cfg = {
        "remove_outliers": "False",
        "save_results": "False",
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
            },
            "Slope0": {
                "type": "slope",
                "ranges": [0, 10, 25, 50, 90],
            },
        },
    }
    # Create StatsProcessing object
    stats_processing = demcompare.StatsProcessing(input_stats_cfg, stats_dem)

    # Test slope layer class 0
    # ----------------------------------------------------------
    classif_class = 0
    dataset_idx = stats_processing.classification_layers_names.index("Slope0")

    # Compute nodata_nan_mask
    no_data_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != no_data_value),
        0,
        stats_processing.dem["image"].data,
    )

    # Get valid class indexes for slope dataset and class 1 mode exclusion
    idxes_map_class_0_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[0][
            classif_class
        ]
        * (
            1
            - (
                stats_processing.classification_layers[
                    dataset_idx
                ].classes_masks[0][classif_class]
                * stats_processing.classification_layers[
                    dataset_idx
                ].classes_masks[1][classif_class]
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
def test_compute_stats_from_cfg():
    """
    Tests the compute_stats. Manually computes
    the stats for the classification_layers for a given class
    and mode and tests that the compute_stats function obtains
    the same values.
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")

    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Initialize sec and ref
    sec = dem_tools.load_dem(cfg["input_sec"]["path"])
    ref = dem_tools.load_dem(
        cfg["input_ref"]["path"],
        classification_layers=(cfg["input_ref"]["classification_layers"]),
    )
    sec, ref, _ = dem_tools.reproject_dems(sec, ref, sampling_source="ref")

    # Compute slope and add it as a classification_layer
    ref = dem_tools.compute_dem_slope(ref)
    sec = dem_tools.compute_dem_slope(sec)
    # Compute altitude diff for stats computation
    stats_dem = dem_tools.compute_alti_diff_for_stats(ref, sec)
    # Initialize stats input configuration
    input_stats_cfg = {
        "remove_outliers": "False",
        "save_results": "False",
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
                "metrics": [
                    {
                        "ratio_above_threshold": {
                            "elevation_threshold": [1, 2, 3]
                        }
                    }
                ],
            },
            "Slope0": {
                "type": "slope",
                "ranges": [0, 10, 25, 50, 90],
                "metrics": ["nmad"],
            },
        },
        "metrics": ["std"],
    }
    # Create StatsProcessing object
    stats_processing = demcompare.StatsProcessing(input_stats_cfg, stats_dem)

    # Compute stats for all classif layers and its metrics
    stats_dataset = stats_processing.compute_stats()
    # ----------------------------------------------------------
    # TEST status layer class 1
    classif_class = 1
    dataset_idx = stats_processing.classification_layers_names.index("Status")
    # Manually compute mean and sum stats

    # Initialize metric objects
    metric_std = Metric("std")
    metric_ratio = Metric(
        "ratio_above_threshold", params={"elevation_threshold": [1, 2, 3]}
    )

    # Compute nodata_nan_mask
    no_data_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != no_data_value),
        0,
        stats_processing.dem["image"].data,
    )
    # Get valid class indexes for status dataset and class 1
    idxes_map_class_1_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[0][
            classif_class
        ]
    )
    # Get class pixels
    classif_map = stats_processing.dem["image"].data[
        idxes_map_class_1_dataset_0
    ]
    # Compute gt metrics
    gt_std = metric_std.compute_metric(classif_map)
    gt_ratio = metric_ratio.compute_metric(classif_map)

    # Get output metrics
    output_std = stats_dataset.get_classification_layer_metric(
        classification_layer="Status",
        classif_class=classif_class,
        mode="standard",
        metric="std",
    )
    output_ratio = stats_dataset.get_classification_layer_metric(
        classification_layer="Status",
        classif_class=classif_class,
        mode="standard",
        metric="ratio_above_threshold",
    )
    # Test that the output metrics are the same as gt
    np.testing.assert_allclose(
        gt_std,
        output_std,
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        gt_ratio,
        output_ratio,
        rtol=1e-02,
    )
    # Test that the nmad metric has not been computed for the Status dataset
    # Test that an error is raised
    with pytest.raises(KeyError):
        stats_dataset.get_classification_layer_metric(
            classification_layer="Status",
            classif_class=classif_class,
            mode="standard",
            metric="nmad",
        )
    # ----------------------------------------------------------
    # TEST Slope layer class 1
    classif_class = 1
    dataset_idx = stats_processing.classification_layers_names.index("Slope0")
    # Manually compute mean and sum stats

    # Initialize metric objects
    metric_std = Metric("std")
    metric_nmad = Metric("nmad")

    # Compute nodata_nan_mask
    no_data_value = stats_processing.classification_layers[dataset_idx].nodata
    nodata_nan_mask = np.apply_along_axis(
        lambda x: (~np.isnan(x)) * (x != no_data_value),
        0,
        stats_processing.dem["image"].data,
    )
    # Get valid class indexes for status dataset and class 1
    idxes_map_class_1_dataset_0 = np.where(
        nodata_nan_mask
        * stats_processing.classification_layers[dataset_idx].classes_masks[0][
            classif_class
        ]
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
