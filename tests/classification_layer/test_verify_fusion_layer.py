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
This module contains functions to test the verify_fusion_layer function.
"""
# pylint:disable = duplicate-code
# Standard imports

# Third party imports
import pytest

# Demcompare imports
from demcompare import dem_tools


@pytest.mark.unit_tests
def test_verify_fusion_layers_sec(initialize_dems_to_fuse):
    """
    Test the verify_fusion_layers function.
    Input data:
    - Two input dems to be fused. The ref dem contains a slope
      classification layer mask, and the sec dem contains a slope
      and a segmentation classification layer mask
    - A manually created classification configuration to fuse
      the sec's slope and segmentation layers
    Validation process:
    - Create the classification configuration
    - Compute the verification using the verify_fusion_layers function
    - Check that the verification is correctly done (no errors are raised)
    - Checked function : dem_tools's verify_fusion_layers
    """
    ref, sec = initialize_dems_to_fuse

    # Test with sec fusion ---------------------------------
    # Initialize stats input configuration
    input_classif_cfg = {
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
        "Fusion0": {"sec": ["Slope0", "Status"], "type": "fusion"},
    }

    dem_tools.verify_fusion_layers(sec, input_classif_cfg, "sec")
    dem_tools.verify_fusion_layers(ref, input_classif_cfg, "ref")


@pytest.mark.unit_tests
def test_verify_fusion_layers_error_ref(initialize_dems_to_fuse):
    """
    Test the verify_fusion_layers function with a non existing
    classification for the ref dem
    Input data:
    - Two input dems to be fused. The ref dem contains a slope
      classification layer mask, and the sec dem contains a slope
      and a segmentation classification layer mask
    - A manually created classification configuration to fuse
      the ref's slope and segmentation layers (the ref segmentation
      does not exist)
    Validation process:
    - Create the classification configuration
    - Compute the verification using the verify_fusion_layers function
    - Check that a ValueError is raised
    - Checked function: dem_tools's verify_fusion_layers
    """
    ref, sec = initialize_dems_to_fuse

    # Test with ref fusion ---------------------------------
    # It should not work as no ref Status exists
    # Initialize stats input configuration
    input_classif_cfg = {
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
        "Fusion0": {"ref": ["Slope0", "Status"], "type": "fusion"},
    }

    dem_tools.verify_fusion_layers(sec, input_classif_cfg, "sec")
    # Test that an error is raised
    with pytest.raises(ValueError):
        dem_tools.verify_fusion_layers(ref, input_classif_cfg, "ref")


@pytest.mark.unit_tests
def test_verify_fusion_layers_cfg_error(initialize_dems_to_fuse):
    """
    Test the verify_fusion_layers function with a non existing
    classification
    Input data:
    - Two input dems to be fused. The ref dem contains a slope
      classification layer mask, and the sec dem contains a slope
      and a segmentation classification layer mask
    - A manually created classification configuration to fuse
      the ref's slope and segmentation layers (the segmentation
      does not exist)
    Validation process:
    - Create the classification configuration
    - Compute the verification using the verify_fusion_layers function
    - Check that a ValueError is raised
    - Checked function: dem_tools's verify_fusion_layers
    """
    ref, _ = initialize_dems_to_fuse

    # Test without defining the Status layer
    # on the input cfg ---------------------------------
    # It should not work as no ref Status exists
    # Initialize stats input configuration
    input_classif_cfg = {
        "Slope0": {
            "type": "slope",
            "ranges": [0, 10, 25, 50, 90],
        },
        "Fusion0": {"ref": ["Slope0", "Status"], "type": "fusion"},
    }

    # Test that an error is raised
    with pytest.raises(ValueError):
        dem_tools.verify_fusion_layers(ref, input_classif_cfg, "ref")
