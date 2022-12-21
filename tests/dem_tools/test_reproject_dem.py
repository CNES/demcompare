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
This module contains functions to test the reproject_dems function.
"""
# pylint:disable = duplicate-code
# Standard imports
import os

# Third party imports
import numpy as np
import pytest

# Demcompare imports
from demcompare import dem_tools
from demcompare.helpers_init import read_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path


@pytest.mark.unit_tests
def test_reproject_dems_sampling_sec(load_gironde_dem):
    """
    Test the reproject_dems function with sampling source sec
    Input data:
    - Ref and sec dems from the "load_gironde_dem" fixture.
    Validation data:
    - Both reprojected dem's attributes when the sampling source
      is sec (both have the sec's resolution and the adapting
      factor is (1,1) since the coregistration offset is obtained
      at the sec's resolution):
      gt_intersection_roi, gt_output_xres, gt_output_yres,
      gt_output_shape, gt_adapting_factor, gt_output_trans
    Validation process:
    - Reproject both dems using the reproject_dems function
    - Check that both reprojected dem's attributes are the same as
      ground truth
    - Checked function : dem_tools's reproject_dems
    """
    sec_orig, ref_orig = load_gironde_dem
    # Reproject dems with sampling source sec  -------------------------------
    reproj_sec, reproj_ref, adapting_factor = dem_tools.reproject_dems(
        sec_orig,
        ref_orig,
        sampling_source=dem_tools.SamplingSourceParameter.SEC.value,
    )

    # Define ground truth values
    gt_intersection_roi = (600255.0, 4990745.0, 689753.076, 5090117.757)
    # Since sampling value is "sec",
    # output resolutions is "sec"'s resolution
    gt_output_yres = -500.00
    gt_output_xres = 500.00
    gt_output_shape = (199, 179)
    gt_adapting_factor = (1.0, 1.0)
    gt_output_trans = np.array(
        [
            6.002550e05,
            5.000000e02,
            0.000000e00,
            5.090245e06,
            0.000000e00,
            -5.000000e02,
        ]
    )

    # Test that both the output dems have the same gt values
    # Tests dems shape
    np.testing.assert_array_equal(
        reproj_sec["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_sec["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(reproj_sec.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_sec.xres, gt_output_xres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_sec.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_sec.georef_transform, gt_output_trans, atol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, atol=1e-02
    )


@pytest.mark.unit_tests
def test_reproject_dems_sampling_ref(load_gironde_dem):
    """
    Test the reproject_dems function with sampling source ref
    Input data:
    - Ref and sec dems from the "load_gironde_dem" fixture.
    Validation data:
    - Both reprojected dem's attributes when the sampling source
      is ref (both have the ref's resolution):
      gt_intersection_roi, gt_output_xres, gt_output_yres,
      gt_output_shape, gt_adapting_factor, gt_output_trans
    Validation process:
    - Reproject both dems using the reproject_dems function
    - Check that both reprojected dem's attributes are the same as
      ground truth
    - Checked function : dem_tools's reproject_dems
    """
    sec_orig, ref_orig = load_gironde_dem
    # Reproject dems with sampling source ref --------------------------------

    reproj_sec, reproj_ref, adapting_factor = dem_tools.reproject_dems(
        sec_orig,
        ref_orig,
        sampling_source=dem_tools.SamplingSourceParameter.REF.value,
    )

    # Define ground truth values
    gt_intersection_roi = (-1.726, 45.039, -0.602, 45.939)
    # Since sampling value is "ref", output resolutions is "ref"'s resolution
    gt_output_yres = -0.0010416
    gt_output_xres = 0.0010416
    gt_output_shape = (865, 1079)
    gt_adapting_factor = (0.199451, 0.190893)
    gt_output_trans = np.array(
        [
            -1.72680447e00,
            1.04166667e-03,
            0.00000000e00,
            4.59394826e01,
            0.00000000e00,
            -1.04166667e-03,
        ]
    )
    # Test that both the output dems have the same gt values
    # Tests dems shape
    np.testing.assert_array_equal(
        reproj_sec["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_sec["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(reproj_sec.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_sec.xres, gt_output_xres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_sec.attrs["bounds"], gt_intersection_roi, rtol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-02
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_sec.georef_transform, gt_output_trans, rtol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, rtol=1e-02
    )


@pytest.mark.unit_tests
def test_reproject_dems_sampling_sec_initial_disparity(load_gironde_dem):
    """
    Test the reproject_dems function with sampling source sec
    and initial disparity
    Input data:
    - Ref and sec dems from the "load_gironde_dem" fixture.
    Validation data:
    - Both reprojected dem's attributes when the sampling source
      is sec (both have the sec's resolution and the adapting
      factor is (1,1) since the coregistration offset is obtained
      at the sec's resolution) and an initial disparity
      for the sec dem is given:
      gt_intersection_roi, gt_output_xres, gt_output_yres,
      gt_output_shape, gt_adapting_factor, gt_output_trans
    Validation process:
    - Reproject both dems using the reproject_dems function
    - Check that both reprojected dem's attributes are the same as
      ground truth
    - Checked function : dem_tools's reproject_dems
    """
    sec_orig, ref_orig = load_gironde_dem

    # Reproject dems with sampling value sec and initial disparity -------------

    reproj_sec, reproj_ref, adapting_factor = dem_tools.reproject_dems(
        sec_orig,
        ref_orig,
        sampling_source=dem_tools.SamplingSourceParameter.SEC.value,
        initial_shift_x=2,
        initial_shift_y=-3,
    )

    # Define ground truth values
    gt_intersection_roi = (601255.0, 4992245.0, 690753.076977, 5091617.757489)
    # Since sampling value is "sec",
    # output resolutions is "sec"'s resolution
    gt_output_yres = -500.00
    gt_output_xres = 500.00
    gt_output_shape = (199, 179)
    gt_adapting_factor = (1.0, 1.0)
    gt_output_trans = np.array(
        [
            6.012550e05,
            5.000000e02,
            0.000000e00,
            5.091745e06,
            0.000000e00,
            -5.000000e02,
        ]
    )

    # Test that both the output dems have the same gt values
    # Tests dems shape
    np.testing.assert_array_equal(
        reproj_sec["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_sec["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(reproj_sec.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_sec.xres, gt_output_xres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_sec.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_sec.georef_transform, gt_output_trans, rtol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, rtol=1e-02
    )


@pytest.mark.unit_tests
def test_reproject_different_z_units():
    """
    Test the reproject_dems function with different
    z units
    Input data:
    - Ref and sec dem present in the "srtm_test_data" test
      data directory. The sec's unit is modified to be cm,
      whilst the ref's unit stays m
    Validation data:
    - Both reprojected dem's attributes when the sampling source
      is sec (both have the sec's resolution):
      gt_intersection_roi, gt_output_xres, gt_output_yres,
      gt_output_shape, gt_adapting_factor, gt_output_trans
    Validation process:
    - Open the strm_test_data's test_config.json file
    - Load the input_ref and input_sec dem using the load_dem function
    - Reproject both dems using the reproject_dems function
    - Check that both reprojected dem's attributes are the same as
      ground truth
    - Checked function : dem_tools's reproject_dems
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Create DEM with cm zunit
    cfg["input_sec"]["path"] = os.path.join(test_data_path, "input/dem_cm.tif")
    cfg["input_sec"]["zunit"] = "cm"

    # Load original dems
    ref_orig = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec_orig = dem_tools.load_dem(cfg["input_sec"]["path"])

    reproj_sec, reproj_ref, adapting_factor = dem_tools.reproject_dems(
        sec_orig, ref_orig
    )

    # Define ground truth values
    gt_intersection_roi = (600255.0, 4990745.0, 689753.076, 5090117.757)
    # Since sampling value is "sec",
    # output resolutions is "sec"'s resolution
    gt_output_yres = -500.00
    gt_output_xres = 500.00
    gt_output_shape = (199, 179)
    gt_adapting_factor = (1.0, 1.0)
    gt_output_trans = np.array(
        [
            6.002550e05,
            5.000000e02,
            0.000000e00,
            5.090245e06,
            0.000000e00,
            -5.000000e02,
        ]
    )

    # Test that both the output dems have the same gt values
    # Tests dems shape
    np.testing.assert_array_equal(
        reproj_sec["image"].shape, reproj_ref["image"].shape
    )
    np.testing.assert_allclose(
        reproj_sec["image"].shape, gt_output_shape, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref["image"].shape, gt_output_shape, rtol=1e-03
    )
    # Tests dems resolution
    np.testing.assert_allclose(reproj_sec.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.yres, gt_output_yres, rtol=1e-02)
    np.testing.assert_allclose(reproj_sec.xres, gt_output_xres, rtol=1e-02)
    np.testing.assert_allclose(reproj_ref.xres, gt_output_xres, rtol=1e-02)
    # Tests dems bounds
    np.testing.assert_allclose(
        reproj_sec.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    np.testing.assert_allclose(
        reproj_ref.attrs["bounds"], gt_intersection_roi, rtol=1e-03
    )
    # Test adapting_factor
    np.testing.assert_allclose(adapting_factor, gt_adapting_factor, rtol=1e-02)
    # Test dems transform
    np.testing.assert_allclose(
        reproj_sec.georef_transform, gt_output_trans, atol=1e-02
    )
    np.testing.assert_allclose(
        reproj_ref.georef_transform, gt_output_trans, atol=1e-02
    )


@pytest.mark.unit_tests
def test_reproject_dems_without_intersection():
    """
    Test that demcompare's reproject_dems function
    raises an error when the input dems
    do not have a common intersection.
    Input data:
    - Ref dem present in the "srtm_test_data" test
      data directory. The sec dem is present in input/reduced_Gironde.tif
      and has no intersection with the ref.
    Validation process:
    - Open the strm_test_data's test_config.json file
    - Load the input_ref and input_sec dem using the load_dem function
    - Check that a NameError is raised when both dems are reprojected
    - Checked function : dem_tools's reproject_dems
    """

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config
    # from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)
    cfg["input_sec"]["path"] = os.path.join(
        test_data_path, "input/reduced_Gironde.tif"
    )

    # get data srtm
    test_data_srtm_path = demcompare_test_data_path("srtm_test_data")
    cfg["input_ref"]["path"] = os.path.join(
        test_data_srtm_path, "input/srtm_ref.tif"
    )

    # Load original dems
    ref_orig = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec_orig = dem_tools.load_dem(cfg["input_sec"]["path"])

    with pytest.raises(NameError) as error_info:
        _, _, _ = dem_tools.reproject_dems(sec_orig, ref_orig)
        assert error_info.value == "ERROR: ROIs do not intersect"
