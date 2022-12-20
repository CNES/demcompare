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
methods in the segmentation classification layer class.
"""

import numpy as np
import pytest


@pytest.mark.unit_tests
def test_create_labelled_map(initialize_segmentation_classification):
    """
    Test the test_create_labelled_map function
    Input data:
    - "test_first_classif" classification layer and its
      classification_layer_mask from
      the "initialize_segmentation_classification" fixture.
    Validation data:
    - The classification_layer_mask present in the input
      dem dataset with the indicator "test_first_classif": gt_map_image
    Validation process:
    - Creation of the input dem dataset
    - Creation of the segmentation classification layer with the
      input dem dataset and the name "test_first_classif"
    - Check that the map_image of the classification layer is the
      classification_layer_mask of the input dem dataset
    - Checked function : ClassificationLayer's _create_labelled_map
    - Checked attribute : ClassificationLayer's map_image

    """

    classif_layer_, classif_data = initialize_segmentation_classification
    # test_create_labelled_map -------------------------------
    # Test that the test_first_classif's map image has been correctly loaded
    # on the dataset
    gt_map_image = classif_data[:, :, 0]
    np.testing.assert_allclose(
        gt_map_image, classif_layer_.map_image["ref"], rtol=1e-02
    )


@pytest.mark.unit_tests
def _test_create_class_masks(initialize_segmentation_classification):
    """
    Test the _create_classification_layer_class_masks function
    Input data:
    - "test_first_classif" classification layer from
      the "initialize_segmentation_classification" fixture.
    Validation data:
    - The manually created classes_masks (one mask per class) corresponding
      to the "test_first_classif" classification layer mask: gt_classes_masks
    Validation process:
    - Creation of the input dem dataset
    - Creation of the segmentation classification layer with the
      input dem dataset and the name "test_first_classif"
    - Check that the classes_masks of the classification layer are the
      same as the gt
    - Checked function : ClassificationLayer's
      _create_classification_layer_class_masks
    - Checked attribute : ClassificationLayer's classes_masks

    """
    classif_layer_, _ = initialize_segmentation_classification

    # test _create_class_masks -------------------------------

    # Create gt classes masks dict for the test_first_classif map
    # The original classes and mask are :
    # "classes": {"sea": [0], "deep_land": [1], "coast": [2], "lake": [3]}
    # classif_data[:, :, 0] = np.array(
    #         [[0, 1, 1], [2, 2, 3], [-9999, 3, 3], [-9999, 1, 0]]
    #     )
    mask_sea = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    mask_deep_land = np.array(
        [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    mask_coast = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    mask_lake = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    )
    gt_classes_masks = {
        "test_first_classif": [mask_sea, mask_deep_land, mask_coast, mask_lake],
    }

    # Test that the computed classes_masks are the same as gt
    np.testing.assert_allclose(
        gt_classes_masks["test_first_classif"],
        classif_layer_.classes_masks["ref"],
        rtol=1e-02,
    )
