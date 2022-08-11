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
# Disable the protected-access to test the functions

"""
This module contains functions to test all the
methods in the Nuth et Kaab coregistration method.
"""
# pylint:disable=protected-access
# pylint:disable=duplicate-code
# Standard imports
import os

# Third party imports
import numpy as np
import pytest
import scipy

# Demcompare imports
from demcompare import coregistration, dem_tools
from demcompare.helpers_init import read_config_file

# Tests helpers
from tests.helpers import demcompare_test_data_path


@pytest.mark.unit_tests
def test_coregister_dems_algorithm():
    """
    Test the _coregister_dems_algorithm function of
    the Nuth et Kaab class.
    Loads the data present in the "gironde_test_data" root data
    directory and test that the output Transform is
    correct.
    The following configurations are tested:
    - "gironde_test_data" test root input DEMs,
     sampling value sec
    - "gironde_test_data" test root input DEMs,
     sampling value ref
    """
    # Test with "gironde_test_data" test root
    # input DEMs and sampling value sec -----------------------------

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Load dems
    ref = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec = dem_tools.load_dem(cfg["input_sec"]["path"])
    sampling_source = "sec"

    # Define ground truth outputs
    rotation = None
    x_offset = -1.4366
    y_offset = -0.4190
    z_offset = -3.4025

    # Reproject and crop DEMs
    (
        reproj_crop_dem,
        reproj_crop_ref,
        _,
    ) = dem_tools.reproject_dems(sec, ref, sampling_source=sampling_source)

    # Coregistration configuration is the following :
    # "coregistration": {
    #    "method_name": "nuth_kaab_internal",
    #    "number_of_iterations": 6,
    #    "estimated_initial_shift_x": 0,
    #    "estimated_initial_shift_y": 0
    # }
    # Create coregistration object
    coregistration_ = coregistration.Coregistration(cfg["coregistration"])
    # Run _coregister_dems_algorithm
    (transform, _, _,) = coregistration_._coregister_dems_algorithm(
        reproj_crop_dem, reproj_crop_ref
    )

    # Test that the outputs match the ground truth
    assert rotation == transform.rotation
    np.testing.assert_allclose(x_offset, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(y_offset, transform.y_offset, rtol=1e-02)
    np.testing.assert_allclose(z_offset, transform.z_offset, rtol=1e-02)

    # Test with "gironde_test_data" test root
    # input DEMs and sampling value ref -----------------------------

    # Get "gironde_test_data" test root data directory absolute path
    test_data_path = demcompare_test_data_path("gironde_test_data")
    # Load "gironde_test_data" demcompare config from input/test_config.json
    test_cfg_path = os.path.join(test_data_path, "input/test_config.json")
    cfg = read_config_file(test_cfg_path)

    # Load dems
    ref = dem_tools.load_dem(cfg["input_ref"]["path"])
    sec = dem_tools.load_dem(cfg["input_sec"]["path"])
    sampling_source = "ref"

    # Define ground truth outputs
    rotation = None
    x_offset = -5.44706
    y_offset = -1.18139
    z_offset = -0.53005

    # Reproject and crop DEMs
    (
        reproj_crop_dem,
        reproj_crop_ref,
        _,
    ) = dem_tools.reproject_dems(sec, ref, sampling_source=sampling_source)

    # Coregistration configuration is the following :
    # "coregistration": {
    #    "method_name": "nuth_kaab_internal",
    #    "number_of_iterations": 6,
    #    "estimated_initial_shift_x": 0,
    #    "estimated_initial_shift_y": 0
    # }
    # Create coregistration object
    coregistration_ = coregistration.Coregistration(cfg["coregistration"])
    # Run coregister_dems
    (transform, _, _,) = coregistration_._coregister_dems_algorithm(
        reproj_crop_dem, reproj_crop_ref
    )

    # Test that the outputs match the ground truth
    assert rotation == transform.rotation
    np.testing.assert_allclose(x_offset, transform.x_offset, rtol=1e-02)
    np.testing.assert_allclose(y_offset, transform.y_offset, rtol=1e-02)
    np.testing.assert_allclose(z_offset, transform.z_offset, rtol=1e-02)


@pytest.mark.unit_tests
def test_grad2d():
    """
    Test the grad2d function
    Manually computes an input array and its
    slope and gradient, and tests that the obtained
    values resulting from the grad2d function are
    correct.

    """
    # Define cfg
    cfg = {
        "method_name": "nuth_kaab_internal",
        "number_of_iterations": 6,
        "estimated_initial_shift_x": 0,
        "estimated_initial_shift_y": 0,
    }
    # Create coregsitration object
    coregistration_ = coregistration.Coregistration(cfg)

    # Define dh array to make the computations
    dh = np.array(
        ([1, 1, 1], [-1, 2, 1], [4, -3, 1], [1, 1, 1], [1, 1, 1]),
        dtype=np.float64,
    )
    # Manually compute gradients
    grad1 = np.array(
        [
            [
                (dh[0][1] - dh[0][0]) / 1,
                (dh[0][2] - dh[0][0]) / 2,
                (dh[0][2] - dh[0][1]) / 1,
            ],
            [
                (dh[1][1] - dh[1][0]) / 1,
                (dh[1][2] - dh[1][0]) / 2,
                (dh[1][2] - dh[1][1]) / 1,
            ],
            [
                (dh[2][1] - dh[2][0]) / 1,
                (dh[2][2] - dh[2][0]) / 2,
                (dh[2][2] - dh[2][1]) / 1,
            ],
            [
                (dh[3][1] - dh[3][0]) / 1,
                (dh[3][2] - dh[3][0]) / 2,
                (dh[3][2] - dh[3][1]) / 1,
            ],
            [
                (dh[4][1] - dh[4][0]) / 1,
                (dh[4][2] - dh[4][0]) / 2,
                (dh[4][2] - dh[4][1]) / 1,
            ],
        ]
    )

    grad2 = np.array(
        [
            [
                (dh[1][0] - dh[0][0]) / 1,
                (dh[1][1] - dh[0][1]) / 1,
                (dh[1][2] - dh[0][2]) / 1,
            ],
            [
                (dh[2][0] - dh[0][0]) / 2,
                (dh[2][1] - dh[0][1]) / 2,
                (dh[2][2] - dh[0][2]) / 2,
            ],
            [
                (dh[3][0] - dh[1][0]) / 2,
                (dh[3][1] - dh[1][1]) / 2,
                (dh[3][2] - dh[1][2]) / 2,
            ],
            [
                (dh[4][0] - dh[2][0]) / 2,
                (dh[4][1] - dh[2][1]) / 2,
                (dh[4][2] - dh[2][2]) / 2,
            ],
            [
                (dh[4][0] - dh[3][0]) / 1,
                (dh[4][1] - dh[3][1]) / 1,
                (dh[4][2] - dh[3][2]) / 1,
            ],
        ]
    )
    # Ground truth slope
    gt_slope = np.sqrt(grad1**2 + grad2**2)
    # Ground truth aspect
    gt_aspect = np.arctan2(-grad1, grad2) + np.pi

    (
        output_slope,
        output_aspect,
    ) = coregistration_._grad2d(dh)
    # Test that the output slope and aspect are the same as ground truth
    np.testing.assert_allclose(output_slope, gt_slope, rtol=1e-02)
    np.testing.assert_allclose(output_aspect, gt_aspect, rtol=1e-02)


@pytest.mark.unit_tests
def test_filter_target():
    """
    Test the filter_target function
    Computes an input target and manually adds noise
    to it, then tests that the filter_target function
    correctly filters the added noise.

    """
    # Initialize cfg
    cfg = {
        "method_name": "nuth_kaab_internal",
        "number_of_iterations": 6,
        "estimated_initial_shift_x": 0,
        "estimated_initial_shift_y": 0,
    }
    # Create coregistration object
    coregistration_ = coregistration.Coregistration(cfg)
    # Define aspect bounds with a np.pi/36 step
    aspect_bounds = np.arange(0, 2 * np.pi, np.pi / 36)
    coregistration_.aspect_bounds = np.arange(0, 2 * np.pi, np.pi / 36)

    # Define target as a sinus array
    target = np.sin(np.array(np.arange(0, 360, 0.05)) * np.pi / 180.0)
    # Define aspect array
    aspect = np.arange(0, 360, 360 / len(target)) * np.pi / 180.0

    noisy_target = []
    gt_filtered_target = []
    gt_slice_filt_median = []

    # Create noisy target by adding noise samples on each slice
    for bounds in aspect_bounds:
        # Slice indexes within aspect
        slice_idxes = np.where(
            (bounds < aspect) & (aspect < bounds + np.pi / 36)
        )
        target_slice = target[slice_idxes]
        # Compute ground truth filtered median with the non-noisy target
        gt_slice_filt_median.append(np.nanmedian(target_slice))
        # Obtain target slice's mean and std
        slice_mean = np.nanmean(target_slice)
        slice_sigma = np.std(target_slice[np.isfinite(target_slice)])
        # Add noise samples that will be outside of the 3 sigma filter
        noisy_target.append(slice_mean + 19 * slice_sigma)
        noisy_target.append(slice_mean - 20 * slice_sigma)
        # The ground truth filtered target has np.nan in the noise sample places
        gt_filtered_target.append(np.nan)
        gt_filtered_target.append(np.nan)
        # Add the non-noisy target_slice samples
        for tar in target_slice:
            noisy_target.append(tar)
            gt_filtered_target.append(tar)
    # Define the noisy_target aspect
    aspect = np.arange(0, 360, 360 / len(noisy_target)) * np.pi / 180.0

    (
        output_slice_filt_median,
        output_filtered_target,
    ) = coregistration_._filter_target(np.array(aspect), np.array(noisy_target))

    # Test that the output filtered target and
    # filtered median are the same as ground truth
    np.testing.assert_allclose(
        gt_filtered_target, output_filtered_target, rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_slice_filt_median, output_slice_filt_median, rtol=1e-02
    )


@pytest.mark.unit_tests
def test_nuth_kaab_single_iter():
    """
    Manually computes an input array and its
    output offsets, and tests that the resulting
    offsets form the nuth_kaab_single_iter are the
    same.

    """
    # Define cfg
    cfg = {
        "method_name": "nuth_kaab_internal",
        "number_of_iterations": 6,
        "estimated_initial_shift_x": 0,
        "estimated_initial_shift_y": 0,
    }
    # Define dh array
    dh = np.array(
        ([1, 1, 1], [-1, 2, 1], [4, -3, 1], [1, 1, 1], [1, 1, 1]),
        dtype=np.float64,
    )
    # Initialize coregistration object
    coregistration_ = coregistration.Coregistration(cfg)
    # Define aspect_bounds attribute as it is defined in the upper function
    aspect_bounds = np.arange(0, 2 * np.pi, np.pi / 36)
    coregistration_.aspect_bounds = np.arange(0, 2 * np.pi, np.pi / 36)

    # Obtain slope and aspect
    slope, aspect = coregistration_._grad2d(dh)
    # Filter slope values below threshold
    slope[np.where(slope < 0.001)] = np.nan
    # Compute target
    target = dh / slope
    # Filer target and aspect non finite values
    target = target[np.isfinite(dh)]
    aspect_filt = aspect[np.isfinite(dh)]
    # Compute filtered target and target median
    (
        slice_filt_median,
        target_filt,
    ) = coregistration_._filter_target(aspect_filt, target)
    # Do least squares optimization with scipy as
    # performed in nuth_kaab_single_iter
    x = aspect_filt.ravel()
    y = target_filt.ravel()
    yf = y[(np.isfinite(x)) & (np.isfinite(y))]
    p0 = (3 * np.std(yf) / (2**0.5), 0, np.mean(yf))

    # least square fit
    def peval(x, p):
        """peval defines the model chosen"""
        return p[0] * np.cos(p[1] - x) + p[2]

    def residuals(p, y, x):
        """residuals function based on peval"""
        err = peval(x, p) - y
        return err

    plsq = scipy.optimize.leastsq(
        residuals,
        p0,
        args=(slice_filt_median, aspect_bounds),
        full_output=1,
    )
    a, b, c = plsq[0]
    # Obtain ground truth offsets
    gt_east = a * np.sin(b)
    gt_north = a * np.cos(b)
    gt_c = c

    (
        output_east,
        output_north,
        output_c,
    ) = coregistration_._nuth_kaab_single_iter(dh, slope, aspect)
    # Test that the output offsets are the same as ground truth
    np.testing.assert_allclose(output_east, gt_east, rtol=1e-02)
    np.testing.assert_allclose(output_north, gt_north, rtol=1e-02)
    np.testing.assert_allclose(output_c, gt_c, rtol=1e-02)


@pytest.mark.unit_tests
def test_interpolate_dem_on_grid():
    """
    Test the interpolate_dem_on_grid function
    Manually computes an input array and its
    spline interpolators, and tests that the resulting
    splines form the interpolate_dem_on_grid are the
    same.
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
        (
            [1, 1, np.nan],
            [-1, 2, 1],
            [4, -3, np.nan],
            [np.nan, 1, 1],
            [1, 1, np.nan],
        ),
        dtype=np.float64,
    )

    # Set target dem grid for interpolation purpose
    xgrid = np.arange(input_dem.shape[1])
    ygrid = np.arange(input_dem.shape[0])

    # Masked input_dem used to obtain spline_1 interpolator
    gt_masked_dem = np.array(
        (
            [1, 1, -9999],
            [-1, 2, 1],
            [4, -3, -9999],
            [-9999, 1, 1],
            [1, 1, -9999],
        ),
        dtype=np.float64,
    )
    # input_dem's mask used to obtain spline_2 interpolators
    gt_nan_mask = np.array(
        (
            [False, False, True],
            [False, False, False],
            [False, False, True],
            [True, False, False],
            [False, False, True],
        ),
        dtype=np.float64,
    )
    output_spline_1, output_spline_2 = coregistration_.interpolate_dem_on_grid(
        input_dem, xgrid, ygrid
    )

    # Compute both ground_truth splines
    gt_spline_1 = scipy.interpolate.RectBivariateSpline(
        ygrid, xgrid, gt_masked_dem, kx=1, ky=1
    )
    gt_spline_2 = scipy.interpolate.RectBivariateSpline(
        ygrid, xgrid, gt_nan_mask, kx=1, ky=1
    )

    # Test that the output splines are the same as ground_truth
    # spline.tck is a tuple (t,c,k) containing the vector of knots,
    # the B-spline coefficients, and the degree of the spline.
    np.testing.assert_allclose(
        gt_spline_1.tck[0], output_spline_1.tck[0], rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_spline_2.tck[0], output_spline_2.tck[0], rtol=1e-02
    )

    np.testing.assert_allclose(
        gt_spline_1.tck[1], output_spline_1.tck[1], rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_spline_2.tck[1], output_spline_2.tck[1], rtol=1e-02
    )

    np.testing.assert_allclose(
        gt_spline_1.tck[2], output_spline_1.tck[2], rtol=1e-02
    )
    np.testing.assert_allclose(
        gt_spline_2.tck[2], output_spline_2.tck[2], rtol=1e-02
    )


@pytest.mark.unit_tests
def test_crop_dem_with_offset():
    """
    Test the crop_dem_with_offset function
    Manually computes an input array and crops it
    with different offsets, and tests that the resulting
    arrays form the crop_dem_with_offset are the
    same.
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

    # Test with positive x_offset and positive y_offset
    x_offset = 2.3
    y_offset = 4.7
    output_cropped_dem = coregistration_.crop_dem_with_offset(
        input_dem, x_offset, y_offset
    )
    gt_cropped_dem = input_dem[
        int(np.floor(y_offset)) : input_dem.shape[0],
        0 : input_dem.shape[1] - int(np.ceil(x_offset)),
    ]
    np.testing.assert_allclose(gt_cropped_dem, output_cropped_dem, rtol=1e-02)

    # Test with positive x_offset and negative y_offset
    x_offset = 2.3
    y_offset = -4.7
    output_cropped_dem = coregistration_.crop_dem_with_offset(
        input_dem, x_offset, y_offset
    )
    gt_cropped_dem = input_dem[
        0 : input_dem.shape[0] - int(np.ceil(-y_offset)),
        0 : input_dem.shape[1] - int(np.ceil(x_offset)),
    ]
    np.testing.assert_allclose(gt_cropped_dem, output_cropped_dem, rtol=1e-02)

    # Test with negative x_offset and positive y_offset
    x_offset = -2.3
    y_offset = 4.7
    output_cropped_dem = coregistration_.crop_dem_with_offset(
        input_dem, x_offset, y_offset
    )
    gt_cropped_dem = input_dem[
        int(np.floor(y_offset)) : input_dem.shape[0],
        int(np.floor(-x_offset)) : input_dem.shape[1],
    ]
    np.testing.assert_allclose(gt_cropped_dem, output_cropped_dem, rtol=1e-02)

    # Test with negative x_offset and negative y_offset
    x_offset = -2.3
    y_offset = -4.7
    output_cropped_dem = coregistration_.crop_dem_with_offset(
        input_dem, x_offset, y_offset
    )
    gt_cropped_dem = input_dem[
        0 : input_dem.shape[0] - int(np.ceil(-y_offset)),
        int(np.floor(-x_offset)) : input_dem.shape[1],
    ]
    np.testing.assert_allclose(gt_cropped_dem, output_cropped_dem, rtol=1e-02)
