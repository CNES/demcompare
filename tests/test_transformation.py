#!/usr/bin/env python
# coding: utf8
# Disable the protected-access to test the functions
# pylint:disable=protected-access
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
This module contains functions to test all the methods
in the transformation class.
"""

# Standard imports
import os

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare import dem_tools, img_tools, transformation
from demcompare.dem_tools import copy_dem
from demcompare.helpers_init import read_config_file

# Tests helpers
from .helpers import demcompare_test_data_path


@pytest.mark.unit_tests
def test_apply():
    """
    Test the apply_transform function
    Creates a DEM xr.Dataset with the georefence
    from the "strm_test_data" test root data directory
    and creates a Transform object. Applies the transform to
    the created DEM to test that the transformation has been
    correctly applied.

    """
    # Get "strm_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("strm_test_data")
    # Load "strm_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    from_dataset = dem_tools.load_dem(cfg["input_sec"]["path"])

    # Define data
    data = np.array(
        [[1, 1, 1], [1, 1, 1], [-1, -32768, 1], [1, 2, -32768], [1, 1, -32768]],
        dtype=np.float32,
    )
    # Create dataset from "strm_test_data" DSM and specific nodata value
    dataset = dem_tools.create_dem(
        data=data,
        transform=from_dataset.georef_transform.data,
        img_crs=from_dataset.crs,
        nodata=-32768,
    )
    # Define pixel offsets
    x_offset = -1.43664
    y_offset = 0.41903
    z_offset = 0.0

    # Create transform object
    transform = transformation.Transformation(
        x_offset=x_offset, y_offset=-y_offset, z_offset=z_offset
    )
    # Copy dataset to compute ground truth transformed dataset
    gt_dataset_transformed = copy_dem(dataset)
    # Add offset to the ground truth dataset
    x_off_coord, y_off_coord = img_tools.convert_pix_to_coord(
        gt_dataset_transformed["georef_transform"].data, -y_offset, x_offset
    )
    gt_dataset_transformed["georef_transform"].data[0] = x_off_coord
    gt_dataset_transformed["georef_transform"].data[3] = y_off_coord

    output_dataset_transformed = transform.apply_transform(dataset)
    # Test that the output_dataset_transformed
    # has the same offsets as the ground truth
    np.testing.assert_allclose(
        output_dataset_transformed.georef_transform,
        gt_dataset_transformed.georef_transform,
        rtol=1e-02,
    )


def test_adapt_transform_offset():
    """
    Test the adapt_transform_offset function
    Creates a Transform object and an adapting factor
    and tests that the offsets has been correctly adapted
    by the input adapting_factor.
    """
    # Define pixel offsets
    x_offset = 30000
    y_offset = 64000
    z_offset = 0.0
    adapting_factor = (0.4, -0.003)
    x_factor, y_factor = adapting_factor
    gt_x_off = x_offset * x_factor
    gt_y_off = y_offset * y_factor

    # Create transform object
    transform = transformation.Transformation(
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
        adapting_factor=adapting_factor,
    )

    # Test that the adapted transform
    # has the same attributes as the ground truth
    np.testing.assert_allclose(
        transform.x_offset,
        gt_x_off,
        rtol=1e-02,
    )
    np.testing.assert_allclose(
        transform.y_offset,
        gt_y_off,
        rtol=1e-02,
    )
