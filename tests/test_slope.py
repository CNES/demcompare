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
This module contains functions to test all the methods in
the slope module.
"""

# Standard imports
import os

# Third party imports
import numpy as np
import pytest
import rasterio

# Demcompare imports
from demcompare import dem_tools, slope
from demcompare.initialization import read_config_file

# Tests helpers
from .helpers import demcompare_test_data_path


@pytest.mark.unit_tests
def test_get_slope():
    """
    Test the get_slope function
    Loads the DEMS present in "gironde_test_data"
    test root data directory and computes the DEM slope
    to test the resulting values.
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "strm_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    from_dataset = dem_tools.load_dem(cfg["input_dem_to_align"]["path"])

    # Generate dsm with the following data and
    # "gironde_test_data" DSM's georef and resolution
    data = np.array([[1, 0, 1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    dem_dataset = dem_tools.create_dem(
        data=data,
        transform=from_dataset.georef_transform.data,
        img_crs=rasterio.crs.CRS.from_epsg(32630),
    )

    # Slope filters are the following
    # conv_x = np.array([[-1, 0, 1],
    #                   [-2, 0, 2],
    #                   [-1, 0, 1]])
    # conv_y = np.array([[-1, -2, -1],
    #                   [ 0,  0,  0],
    #                   [ 1,  2,  1]])

    # Convolution result
    gx = np.array([[4, 0, -4], [2, -2, -4], [-2, -6, -4]])
    gy = np.array([[0, 0, 0], [6, 2, 0], [6, 2, 0]])
    # Dataset's absolute resolutions are 500, 500
    distx = np.abs(dem_dataset.xres)
    disty = np.abs(dem_dataset.yres)

    # Compute tan(slope) and aspect
    tan_slope = np.sqrt((gx / distx) ** 2 + (gy / disty) ** 2) / 8
    gt_slope = np.arctan(tan_slope) * 100

    output_slope = slope.get_slope(dem_dataset)
    # Test that the output_slope is the same as ground_truth
    np.testing.assert_allclose(gt_slope, output_slope, rtol=1e-02)
