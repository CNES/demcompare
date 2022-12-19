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
methods in the StatsPair class.
"""
# pylint:disable = duplicate-code
# Standard imports

import numpy as np

# Third party imports
import pytest


@pytest.mark.unit_tests
def test_add_classif_layer_and_mode_stats_names(initialize_stats_dataset):
    """
    Test the add_classif_layer_and_mode_stats function for
    the classification layer names
    Input data:
    - StatsDataset from the "initialize_stats_dataset" fixture
    Validation data:
    - Ground truth names of the datasets that should
      be present on StatsDataset: gt_dataset_names
    Validation process:
    - Check that the classification layers on StatsDataset
      are the same as ground truth
    - Checked function : StatsDataset's
      add_classif_layer_and_mode_stats
    """
    stats_dataset = initialize_stats_dataset

    # Test dataset_names attribute
    gt_dataset_names = ["Status", "Slope0"]
    assert (
        list(stats_dataset.classif_layers_and_modes.keys()) == gt_dataset_names
    )


@pytest.mark.unit_tests
@pytest.mark.functional_tests
def test_add_classif_layer_and_mode_stats_status_layer(
    initialize_stats_dataset, input_stats_status_results, input_images
):
    """
    Test the add_classif_layer_and_mode_stats function for
    the Status classification layer
    Input data:
    - StatsDataset from the "initialize_stats_dataset" fixture
    Validation data:
    - Ground truth information and metrics of the classification
      layer: gt_image, input_stats_status_standard,
      input_stats_status_intersection
    Validation process:
    - Check that the metrics "mean" and "nbpts" for class 0 and 1
      and modes standard and intersection on the StatsDataset
      are the same as ground truth
    - Check that the image and the image_by_class for standard
      and intersection are the same as ground truth
    - Checked function : StatsDataset's
      add_classif_layer_and_mode_stats and get_classification_layer_dataset
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
@pytest.mark.functional_tests
def test_add_classif_layer_and_mode_stats_slope_layer(
    initialize_stats_dataset, input_stats_slope_results, input_images
):
    """
    Test the add_classif_layer_and_mode_stats function for
    the Slope0 classification layer
    Input data:
    - StatsDataset from the "initialize_stats_dataset" fixture
    Validation data:
    - Ground truth information and metrics of the classification
      layer: gt_image, input_stats_status_standard,
      input_stats_status_intersection, nput_stats_status_exclusion
    Validation process:
    - Check that the metrics "mean" and "nbpts" for class 0 and 1
      and modes standard, intersection and exclusion on the StatsDataset
      are the same as ground truth
    - Check that the image and the image_by_class for standard
      and intersection are the same as ground truth
    - Checked function : StatsDataset's
      add_classif_layer_and_mode_stats and get_classification_layer_dataset
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
