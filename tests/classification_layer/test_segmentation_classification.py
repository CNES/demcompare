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
methods in the segmentation classification layer class.
"""

import numpy as np
import pytest
import xarray as xr

# Demcompare imports
from demcompare.classification_layer.classification_layer import (
    ClassificationLayer,
)


@pytest.mark.unit_tests
def test_create_classification_layer_class_masks():
    """
    Test the _create_classification_layer_class_masks function
    Manually computes an input dem and with two input
    classification layers,
    and then creates the first classification layer object
    and verifies the computed map_image (function _create_labelled_map)
    and then verifies the computed sets_masks_dict
    (function _create_class_masks)
    """
    # Classification layer configuration
    layer_name = "test_first_classif"
    clayer = {
        "type": "segmentation",
        "classes": {"sea": [0], "deep_land": [1], "coast": [2], "lake": [3]},
        "save_results": False,
        "output_dir": "",
        "no_data": -9999,
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
    dataset["classification_layers"] = xr.DataArray(
        data=classif_data,
        coords=coords_classification_layers,
        dims=["row", "col", "indicator"],
    )

    # Initialize classification layer object
    classif_layer_ = ClassificationLayer(
        name=layer_name,
        classification_layer_kind=clayer["type"],
        dem=dataset,
        cfg=clayer,
    )

    # test_create_labelled_map -------------------------------
    # Test that the test_first_classif's map image has been correctly loaded
    # on the dataset
    gt_map_image = classif_data[:, :, 0]
    np.testing.assert_allclose(
        gt_map_image, classif_layer_.map_image[-1], rtol=1e-02
    )

    # test _create_class_masks -------------------------------

    # Create gt sets masks dict for the test_first_classif map
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
        "Slope0": [mask_sea, mask_deep_land, mask_coast, mask_lake],
    }

    # Test that the computed sets_masks_dict is the same as gt
    np.testing.assert_allclose(
        gt_classes_masks["Slope0"],
        classif_layer_.classes_masks[-1],
        rtol=1e-02,
    )
