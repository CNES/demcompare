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
This module contains functions to test the load_dem function.
"""
# pylint:disable = duplicate-code
# Standard imports
import os
from tempfile import TemporaryDirectory

# Third party imports
import numpy as np
import pytest
import rasterio

# Demcompare imports
from demcompare import dem_tools, load_input_dems
from demcompare.helpers_init import read_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path, temporary_dir


@pytest.mark.unit_tests
def test_load_dem():
    """
    Test the load_dem function
    Input data:
    - Ref dem present in the "srtm_test_data" test
      data directory.
    Validation data:
    - The dem's ground truth attributes: gt_nodata,
      gt_xres, gt_yres, gt_plani_unit, gt_zunit, gt_georef,
      gt_shape, gt_transform
    Validation process:
    - Open the strm_test_data's test_config.json file
    - Load the input_ref dem using the load_dem function
    - Check that the attributes of the input dem are the
      same as ground truth
    - Checked function : dem_tools's load_dem
    """
    # Get "srtm_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("srtm_test_data")
    # Load "srtm_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # Load dem
    dem = dem_tools.load_dem(cfg["input_ref"]["path"])

    # Define ground truth values
    gt_nodata = -32768.0
    gt_xres = 0.000833333333333333
    gt_yres = -0.0008333333333333334
    gt_plani_unit = "deg"
    gt_zunit = "m"
    gt_georef = "EPSG:4326"
    gt_shape = (1000, 1000)
    gt_transform = np.array(
        [
            4.00000000e01,
            8.33333333e-04,
            0.00000000e00,
            4.00000000e01,
            0.00000000e00,
            -8.33333333e-04,
        ]
    )
    # Test that the loaded dem has the groud truth values
    assert gt_nodata == dem.attrs["nodata"]
    np.testing.assert_allclose(gt_xres, dem.attrs["xres"], rtol=1e-02)
    np.testing.assert_allclose(gt_yres, dem.attrs["yres"], rtol=1e-02)
    assert gt_plani_unit == dem.attrs["plani_unit"]
    assert gt_zunit == dem.attrs["zunit"]
    assert gt_georef == dem.attrs["crs"]
    assert gt_shape == dem["image"].shape
    np.testing.assert_allclose(gt_transform, dem.georef_transform, rtol=1e-02)


# Filter warning: Dataset has no geotransform, gcps, or rpcs
@pytest.mark.filterwarnings(
    "ignore: Dataset has no geotransform, gcps, or rpcs"
)
@pytest.mark.unit_tests
def test_load_dem_with_nodata():
    """
    Test the load_dem function with NODATA values
    Input data:
     - Hand crafted dem with -32768.0 value as NODATA value
    Validation data:
    - NONE
    Validation process:
    - Load input/test_config.json
    - Replace input_sec by input_ref in config
    - Test that ValueError is raised
    """

    # Same dem for second and reference
    # Create temporary directory for test output
    with TemporaryDirectory(dir=temporary_dir()) as tmp_dir:
        dem_nodata = np.ones((20, 20)) * -32768.0

        new_dataset = rasterio.open(
            f"{tmp_dir}/dem_nodata.tif",
            "w",
            driver="GTiff",
            height=dem_nodata.shape[0],
            width=dem_nodata.shape[1],
            count=1,
            dtype=str(dem_nodata.dtype),
        )
        new_dataset.write(dem_nodata, 1)
        new_dataset.close()

        # Test that ValueError is raised
        with pytest.raises(ValueError):
            _ = dem_tools.load_dem(f"{tmp_dir}/dem_nodata.tif", nodata=-32768.0)


@pytest.mark.unit_tests
def test_load_dem_with_roi_image_coords():
    """
    Test the load_dem function with input ROI in
    pixel format
    Input data:
    - input DEMs present in "strm_test_data" test root data directory
    Validation data:
    - Manually computed ROI and BoundingBox obtained by rasterio
    Validation process:
    - Load the dem with the load_dem function.
    - Verify that the loaded dem has a correct ROI
    - Checked function: load_dem
    - Checked attribute: dem's "bounds"
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # add roi to cfg
    cfg["input_ref"]["roi"] = {"x": 0, "y": 0, "w": 500, "h": 500}
    ref, _ = load_input_dems(cfg)

    # Instantiate ground truth ROI thanks to rasterio
    src_dem = rasterio.open(cfg["input_ref"]["path"])
    window_dem = rasterio.windows.Window(
        0,
        0,
        500,
        500,
    )
    left, bottom, right, top = rasterio.windows.bounds(
        window_dem, src_dem.transform
    )
    gt_roi = rasterio.coords.BoundingBox(left, bottom, right, top)

    assert gt_roi == ref.attrs["bounds"]


@pytest.mark.unit_tests
def test_load_dem_wrong_classification_map():
    """
    Test that the load_dem function
    raises an error when given a
    classification layer map path
    that has different dimensions
    than its support.
    Input data:
    - path to the input DEM present in "gironde_test_data"
      test root data directory
    - classification layer dictionary with a mask which is
      smaller than the intput dem
    Validation process:
    - Load the dem with the load_dem function and the
      classification layer dictionary
    - Check that a KeyError is raised
    - Checked function: load_dem
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Get different size classification layer
    classif_layer_path = os.path.join(
        test_data_path,
        "input/Small_FinalWaveBathymetry_T30TXR_20200622T105631_Status.TIF",
    )
    cfg["input_sec"]["classification_layers"] = classif_layer_path

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
        }
    }

    # Test that an error is raised
    with pytest.raises(KeyError):
        _ = dem_tools.load_dem(
            cfg["input_ref"]["path"], classification_layers=input_classif_cfg
        )


@pytest.mark.unit_tests
def test_load_dem_with_roi_ground_coords():
    """
    Test the load_dem function with input ROI in
    coordinate format
    Input data:
    - input DEMs present in "strm_test_data" test root data directory
    Validation data:
    - Manually computed ROI and BoundingBox obtained by rasterio
    Validation process:
    - Load the dem with the load_dem function.
    - Verify that the loaded dem has a correct ROI
    - Checked function: load_dem
    - Checked attribute: dem's "bounds"
    """
    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    # add roi to cfg
    cfg["input_ref"]["roi"] = {
        "left": -1.7413878040009774,
        "bottom": 45.41864925736226,
        "right": -1.2205544690009773,
        "top": 45.93948259236226,
    }
    ref, _ = load_input_dems(cfg)

    # Instantiate ground truth ROI thanks to rasterio
    gt_roi = rasterio.coords.BoundingBox(
        -1.7413878040009774,
        45.41864925736226,
        -1.2205544690009773,
        45.93948259236226,
    )

    assert gt_roi == ref.attrs["bounds"]
