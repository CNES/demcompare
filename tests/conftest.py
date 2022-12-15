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
This module contains fixtures
"""

# pylint:disable = duplicate-code
# pylint:disable = too-many-lines
# Standard imports
import os

import numpy as np
import pytest
import xarray as xr

# Demcompare imports
import demcompare
from demcompare import coregistration, dem_tools
from demcompare.classification_layer.classification_layer import (
    ClassificationLayer,
)
from demcompare.classification_layer.fusion_classification import (
    FusionClassificationLayer,
)
from demcompare.helpers_init import read_config_file
from demcompare.stats_dataset import StatsDataset
from demcompare.stats_processing import StatsProcessing

# Tests helpers
from .helpers import demcompare_test_data_path


@pytest.fixture
def initialize_stats_processing():
    """
    Fixture to initialize the stats_processing object for tests
    - Loads the ref and sec dems present in the "gironde_test_data_sampling_ref"
      test data directory using the load_dem function
    - Reprojects both dems using the reproject_dems function
    - Computes the slope of both reprojected dems
    - Computes the altitude difference between both dems
    - Computes an input statistics configuration containing a slope
      classification layer called "Slope0" and a segmentation
      classification layer called "Status"
    - Creates a StatsProcessing object with the altitude difference and
      the statistics configuration
    - Returns the StatsProcessing object
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

    return stats_processing


@pytest.fixture
def initialize_stats_processing_with_metrics():
    """
    Fixture to initialize the stats_processing object
    with an input cfg containing the desired metrics for tests
    - Loads the ref and sec dems present in the "gironde_test_data_sampling_ref"
      test data directory using the load_dem function
    - Reprojects both dems using the reproject_dems function
    - Computes the slope of both reprojected dems
    - Computes the altitude difference between both dems
    - Computes an input statistics configuration containing a slope
      classification layer called "Slope0" and a segmentation
      classification layer called "Status", each layer having
      some specified metrics
    - Creates a StatsProcessing object with the altitude difference and
      the statistics configuration
    - Returns the StatsProcessing object
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

    return stats_processing


@pytest.fixture
def initialize_dems_to_fuse():
    """
    Fixture to initialize two dems to be fused
    - Loads the ref and sec dems present in the "gironde_test_data"
      test data directory using the load_dem function, the sec dem
      containing a segmentation classification mask
    - Reprojects both dems using the reproject_dems function
    - Computes the slope of each dem
    - Returns both dems
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
    return ref, sec


@pytest.fixture
def load_gironde_dem():
    """
    Fixture to initialize the gironde dem for tests
    - Loads the ref and sec dems present in the "gironde_test_data"
    - Returns both dems
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load original dems
    ref_orig = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec_orig = dem_tools.load_dem(cfg["input_sec"]["path"])

    return sec_orig, ref_orig


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


@pytest.fixture
def initialize_stats_dataset(
    input_stats_status_results, input_stats_slope_results, input_images
):
    """
    Fixture to initialize the StatsDataset object for tests
    The StatsDataset contains the metrics and information of
    a segmentation layer named "Status" for mode
    standard and intersection, and a slope layer named
    "Slope0" for mode standard, intersection and exclusion
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


@pytest.fixture()
def initialize_transformation():
    """
    Fixture to initialize the image georef transform
    - Creates and returns a transformation array
    """
    # Define transformation
    trans = np.array(
        [
            5.962550e05,
            5.000000e02,
            0.000000e00,
            5.099745e06,
            0.000000e00,
            -5.000000e02,
        ]
    )
    return trans


@pytest.fixture
def initialize_dem_and_coreg():
    """
    Fixture to initialize the input dem and the
    coregistration object
    """

    # Define cfg
    cfg = {
        "method_name": "nuth_kaab_internal",
        "number_of_iterations": 6,
        "estimated_initial_shift_x": 0,
        "estimated_initial_shift_y": 0,
    }

    # Initialize coregistration object
    coregistration_ = coregistration.Coregistration(cfg)

    # Define input_dem array
    input_dem = np.array(
        ([1, 1, 1], [-1, 2, 1], [4, -3, 2], [2, 1, 1], [1, 1, 2]),
        dtype=np.float64,
    )
    return coregistration_, input_dem


@pytest.fixture(name="initialize_slope_layer")
def fixture_initialize_slope_layer():
    """
    Fixture to initialize a slope layer

    - Manually creates input dem dataset
    - Computes the dem slope with the "compute_dem_slope" function
    - Creates a slope classification layer created using the input
      dataset. The classification layer is called "Slope0"
    - Returns the created classification layer object and the input dem

    """
    # Generate dsm with the following data and
    # "gironde_test_data" DSM's georef and resolution
    data = np.array([[1, 0, 1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    dem_dataset = dem_tools.create_dem(data=data)
    # Compute dem's slope
    dem_dataset = dem_tools.compute_dem_slope(dem_dataset)
    # Classification layer configuration
    layer_name = "Slope0"
    clayer = {
        "type": "slope",
        "ranges": [0, 5, 10, 25, 45],
        "output_dir": "",
        "metrics": ["mean"],
    }

    # Initialize slope classification layer object
    slope_classif_layer_ = ClassificationLayer(
        name=layer_name,
        classification_layer_kind=str(clayer["type"]),
        dem=dem_dataset,
        cfg=clayer,
    )
    return slope_classif_layer_, dem_dataset


@pytest.fixture
def initialize_segmentation_classification():
    """
    Fixture to initialize a segmentation classification

    - Manually creates input dem dataset with an image array and
      two classification_layer_masks, called "test_first_classif" and
      "test_second_classif"
    - Creates a segmentation classification layer created using the input
      dataset. The classification layer is called "test_first_classif".
    - Returns the created classification layer object and the
      classification layer masks present in the input dem.
    """
    # Classification layer configuration
    layer_name = "test_first_classif"
    clayer = {
        "type": "segmentation",
        "classes": {"sea": [0], "deep_land": [1], "coast": [2], "lake": [3]},
        "output_dir": "",
        "nodata": -9999,
        "metrics": ["mean"],
    }
    # Create the dems data
    data = np.array(
        [[59, 59, 59], [46, 46, 46], [-9999, 38, 83], [-9999, 31, 67]],
        dtype=np.float32,
    )

    # Create the dataset
    dataset = xr.Dataset(
        {"image": (["row", "col"], data.astype(np.float32))},
        coords={
            "row": np.arange(data.shape[0]),
            "col": np.arange(data.shape[1]),
        },
    )
    # Initialize the data of the classification layers
    classif_data = np.full((data.shape[0], data.shape[1], 2), np.nan)
    classif_data[:, :, 0] = np.array(
        [[0, 1, 1], [2, 2, 3], [-9999, 3, 3], [-9999, 1, 0]]
    )
    classif_data[:, :, 1] = np.array(
        [[0, 4, 4], [6, 6, 6], [-9999, -9999, 0], [-9999, 0, 0]]
    )

    classif_name = ["test_first_classif", "test_second_classif"]

    # Initialize the coordinates of the classification layers
    coords_classification_layers = [
        dataset.coords["row"],
        dataset.coords["col"],
        classif_name,
    ]
    # Add the datarray with input classification layers
    # to the dataset
    dataset["classification_layer_masks"] = xr.DataArray(
        data=classif_data,
        coords=coords_classification_layers,
        dims=["row", "col", "indicator"],
    )

    # Initialize classification layer object
    classif_layer_ = ClassificationLayer(
        name=layer_name,
        classification_layer_kind=str(clayer["type"]),
        dem=dataset,
        cfg=clayer,
    )

    return classif_layer_, classif_data


@pytest.fixture
def initialize_fusion_layer():
    """
    Fixture to initialize the fusion layer
    - Manually creates two input dems (dataset_ref and dataset_sec)
      with an image array
    - The dataset_sec dem contains
      one classification_layer_mask, called "seg_classif"
    - Computes the slope of both dems using the "compute_dem_slope" function
    - Computes the altitude difference between both dems
    - Creates a segmentation classification layer created using
      altitude difference dataset
      The classification layer is called "seg_classif".
    - Creates a slope classification layer using the altitude difference
      dataset
    - Forces the map_image["sec"] of the slope classification to be
      a manually created array
    - Creates a Fusion classification layer using the segmentation and
      the slope classification layers with sec support
    - Returns the created fusion classification layer object
    """
    data = np.array(
        [[0, 1, 1], [0, -9999, 1], [-9999, 1, 1], [-9999, 1, 1]],
        dtype=np.float32,
    )

    # Classification layer configuration
    seg_name = "seg_classif"
    seg_clayer = {
        "type": "segmentation",
        "classes": {"sea": [0], "deep_land": [1]},
        "output_dir": "",
        "nodata": -9999,
        "metrics": ["mean"],
    }

    # Create the ref dataset
    dataset_ref = dem_tools.create_dem(data=data)

    # Create the sec dataset

    # Initialize the data of the classification layers
    classif_data = np.full((data.shape[0], data.shape[1], 1), np.nan)
    classif_data[:, :, 0] = np.array(
        [[0, 1, 1], [1, 1, 0], [-9999, 1, 1], [-9999, 1, 0]]
    )
    classif_name = ["seg_classif"]

    # Initialize the coordinates of the classification layers
    coords_classification_layers = [
        dataset_ref.coords["row"],
        dataset_ref.coords["col"],
        classif_name,
    ]
    # Create the datarray with input classification layers
    # for the sec dataset
    seg_classif_layer_mask = xr.DataArray(
        data=classif_data,
        coords=coords_classification_layers,
        dims=["row", "col", "indicator"],
    )

    # Create the sec dataset
    dataset_sec = dem_tools.create_dem(
        data=data, classification_layer_masks=seg_classif_layer_mask
    )

    # Compute slope and add it as a classification_layer
    dataset_ref = dem_tools.compute_dem_slope(dataset_ref)
    dataset_sec = dem_tools.compute_dem_slope(dataset_sec)
    # Compute altitude diff
    final_altitude_diff = dem_tools.compute_alti_diff_for_stats(
        dataset_ref, dataset_sec
    )

    # Initialize classification layer object
    seg_classif_layer_ = ClassificationLayer(
        name=seg_name,
        classification_layer_kind=str(seg_clayer["type"]),
        dem=final_altitude_diff,
        cfg=seg_clayer,
    )

    # Classification layer configuration
    slope_name = "Slope0"
    slope_clayer = {
        "type": "slope",
        "ranges": [0, 5, 10],
        "output_dir": "",
        "nodata": -9999,
        "metrics": ["mean"],
    }

    # Initialize slope classification layer object
    slope0_classif_layer_ = ClassificationLayer(
        name=slope_name,
        classification_layer_kind=str(slope_clayer["type"]),
        dem=final_altitude_diff,
        cfg=slope_clayer,
    )

    # Initialize ground truth sec map_image.
    # input ranges are [0, 5, 10, 25, 45]
    slope_dem_map_img = np.array(
        [
            [0.0, 0.0, 5.0],
            [5.0, 10.0, 10.0],
            [0.0, -9999, -9999],
            [0.0, 10.0, 5.0],
        ]
    )
    # Set ground truth map_image on the classification_layer object
    # It will be placed on idx 1
    slope0_classif_layer_.map_image["sec"] = slope_dem_map_img
    # Create sets_masks on the input map_image
    slope0_classif_layer_._create_class_masks()
    # Create a fusion layer with the two input classes
    # Since we want to fusion sec images, map_idx is 1
    fusion_layer_ = FusionClassificationLayer(
        [seg_classif_layer_, slope0_classif_layer_],
        support="sec",
        name="Fusion0",
        metrics=["mean"],
    )
    return fusion_layer_


@pytest.fixture
def get_default_metrics():
    """
    Fixture to initialize _DEFAULT_METRICS list
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")

    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    test_cfg = read_config_file(test_cfg_path)

    statsprocessing_ = StatsProcessing(cfg=test_cfg["statistics"])

    metrics_list = statsprocessing_._DEFAULT_METRICS["metrics"]

    return metrics_list
