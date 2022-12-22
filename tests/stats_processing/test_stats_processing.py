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
methods in the StatsProcessing class.
"""
import glob

# pylint:disable = duplicate-code
# Standard imports
import os
from tempfile import TemporaryDirectory

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
from demcompare.helpers_init import mkdir_p, read_config_file, save_config_file
from demcompare.metric import Metric

# Tests helpers
from tests.helpers import demcompare_test_data_path, temporary_dir


@pytest.mark.unit_tests
def test_create_classif_layers():
    """
    Test the create_classif_layers function
    Input data:
    - Ref and sec dems present in the "gironde_test_data" test data
      directory
    - Manually computed statistics configuration containing a slope,
      a segmentation and a fusion classification layer and
      some metrics at different configuration levels
    Validation data:
    - The ground truth configuration of the ClassificationLayers
      to be created by StatsProcessing: gt_status_layer,
      gt_slope_layer, gt_fusion_layer, gt_global_layer
    Validation process:
    - Loads the ref and sec dems present in the "gironde_test_data"
      test data directory using the load_dem function
    - Reprojects both dems using the reproject_dems function
    - Computes the slope of both reprojected dems
    - Computes the altitude difference between both dems
    - Computes an input statistics configuration containing a slope,
      a segmentation and a fusion layer
    - Creates a StatsProcessing object with the altitude difference and
      the statistics configuration
    - Checks that the configuration of the ClassificationLayers
      created in StatsProcessing are the same as ground truth
    - Checked function: StatsProcessing's
      _create_classif_layers
    - Checked attribute: StatsProcessing's classification_layers
      configuration
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
                "metrics": ["nmad"],
            },
            "Fusion0": {
                "type": "fusion",
                "sec": ["Slope0", "Status"],
                "metrics": ["sum"],
            },
        },
        "metrics": [
            "mean",
            {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}},
        ],
    }
    # Create StatsProcessing object
    stats_processing = demcompare.StatsProcessing(input_stats_cfg, stats_dem)

    # Create ground truth Status layer
    gt_status_layer = ClassificationLayer(
        "Status",
        "segmentation",
        {
            "type": "segmentation",
            "classes": {
                "valid": [0],
                "KO": [1],
                "Land": [2],
                "NoData": [3],
                "Outside_detector": [4],
            },
            "metrics": [
                "mean",
                {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}},
            ],
        },
        stats_dem,
    )
    # Create ground truth Slope layer
    gt_slope_layer = ClassificationLayer(
        "Slope0",
        "slope",
        {
            "type": "slope",
            "ranges": [0, 10, 25, 50, 90],
            "metrics": [
                "nmad",
                "mean",
                {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}},
            ],
        },
        stats_dem,
    )
    # Create ground truth Global layer
    gt_global_layer = ClassificationLayer(
        "global",
        "global",
        {
            "type": "global",
            "metrics": [
                "mean",
                {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}},
            ],
        },
        stats_dem,
    )
    # Create ground truth Fusion layer
    gt_fusion_layer = FusionClassificationLayer(
        [gt_status_layer, gt_slope_layer],
        support="sec",
        name="Fusion0",
        metrics=[
            "sum",
            "mean",
            {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}},
        ],
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
    with an input configuration
    that does not specify any classification layer
    Input data:
    - Sec dem present in the "gironde_test_data" test data
      directory
    - Manually computed statistics configuration that
      does not contain any classification layer
    Validation data:
    - The ground truth configuration of the ClassificationLayer
      to be created by StatsProcessing: gt_global_layer
    Validation process:
    - Loads the sec dem present in the "gironde_test_data"
      test data directory using the load_dem function
    - Computes an input statistics configuration
    - Creates a StatsProcessing object with the input dem and
      the statistics configuration
    - Checks that the configuration of the ClassificationLayer
      created in StatsProcessing is the same as ground truth
    - Checked function: StatsProcessing's
      _create_classif_layers
    - Checked attribute: StatsProcessing's classification_layers
      configuration
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

    # Initialize stats input configuration
    input_stats_cfg = {
        "remove_outliers": "False",
    }

    # Create StatsProcessing object
    stats_processing = demcompare.StatsProcessing(input_stats_cfg, sec)

    # Create ground truth default Global layer
    gt_global_layer = ClassificationLayer(
        "global",
        "global",
        {
            "type": "global",
            "metrics": [
                "mean",
                "median",
                "max",
                "min",
                "sum",
                "squared_sum",
                "std",
            ],
        },
        sec,
    )

    # Get StatsProcessing created classification layers
    output_classification_layers = stats_processing.classification_layers
    # Test that StatsProcessing's classification layer's cfg is the same
    # as the gt layer's cfg
    assert output_classification_layers[0].cfg == gt_global_layer.cfg


@pytest.mark.unit_tests
@pytest.mark.functional_tests
def test_compute_stats_from_cfg_status(
    initialize_stats_processing_with_metrics,
):
    """
    Tests that the initialization of StatsProcessing
    with an input configuration containing metrics correctly
    computes the specified metrics for
    the segmentation classification layer
    Input data:
    - StatsProcessing object from the
      "initialize_stats_processing_with_metrics" fixture
    Validation data:
    - The manually computed statistics on the pixels
      corresponding to the class 1 of the segmentation classification
      layer: gt_std, gt_ratio
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
    - Checks that the nmad metric has not been computed
    - Checked function: StatsProcessing's initialization,
      compute_stats and get_classification_layer_metric
    """
    # Initialize stats processing object with metrics on the cfg
    stats_processing = initialize_stats_processing_with_metrics

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


# Filter warning: Assigning the 'data' attribute will be removed in the future
@pytest.mark.filterwarnings("ignore: Assigning the 'data' attribute")
@pytest.mark.unit_tests
@pytest.mark.functional_tests
def test_statistics_output_dir():
    """
    Test that demcompare's execution with
    the statistics output_dir parameter
    set correctly saves to disk all
    classification layer's maps, csv and json files.
    Input data:
    - Configuration file on the "gironde_test_data_sampling_ref"
      directory
    Validation data:
    - The manually computed list of files to be present
      on the output directory after the statistics computation
    Validation process:
    - Reads the input configuration file and suppress the
      coregistration configuration
    - Run demcompare
    - Check that the output files are present on the output directory
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data_sampling_ref")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # clean useless coregistration step
    cfg.pop("coregistration")

    gt_truth_list_for_fusion_global_status = [
        "ref_rectified_support_map.tif",
        "stats_results.csv",
        "stats_results.json",
    ]

    gt_truth_list_for_slope = [
        "ref_rectified_support_map.tif",
        "sec_rectified_support_map.tif",
        "stats_results.csv",
        "stats_results.json",
        "stats_results_exclusion.csv",
        "stats_results_exclusion.json",
        "stats_results_intersection.csv",
        "stats_results_intersection.json",
    ]

    # Test with statistics output_dir set
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        mkdir_p(tmp_dir)
        # Modify test's output dir in configuration to tmp test dir
        cfg["output_dir"] = tmp_dir

        # Set a new test_config tmp file path
        tmp_cfg_file = os.path.join(tmp_dir, "test_config.json")

        # Save the new configuration inside the tmp dir
        save_config_file(tmp_cfg_file, cfg)

        # Run demcompare with "gironde_test_data_sampling_ref"
        # Put output_dir in coregistration dict config
        demcompare.run(tmp_cfg_file)

        assert os.path.isfile(tmp_dir + "/stats/dem_for_stats.tif") is True

        assert os.path.exists(tmp_dir + "/stats/Fusion0/") is True
        list_basename = [
            os.path.basename(x) for x in glob.glob(tmp_dir + "/stats/Fusion0/*")
        ]
        assert (
            all(
                file in list_basename
                for file in gt_truth_list_for_fusion_global_status
            )
            is True
        )

        assert os.path.exists(tmp_dir + "/stats/global/") is True
        list_basename = [
            os.path.basename(x) for x in glob.glob(tmp_dir + "/stats/global/*")
        ]
        assert (
            all(
                file in list_basename
                for file in gt_truth_list_for_fusion_global_status
            )
            is True
        )

        assert os.path.exists(tmp_dir + "/stats/Slope0/") is True
        list_basename = [
            os.path.basename(x) for x in glob.glob(tmp_dir + "/stats/Slope0/*")
        ]
        assert (
            all(file in list_basename for file in gt_truth_list_for_slope)
            is True
        )

        assert os.path.exists(tmp_dir + "/stats/Status/") is True
        list_basename = [
            os.path.basename(x) for x in glob.glob(tmp_dir + "/stats/Status/*")
        ]
        assert (
            all(
                file in list_basename
                for file in gt_truth_list_for_fusion_global_status
            )
            is True
        )
