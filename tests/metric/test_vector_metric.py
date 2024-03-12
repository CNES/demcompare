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
methods in the vector metric class.
"""
# pylint:disable=protected-access
# Third party imports
import numpy as np
import pytest

from demcompare.metric import Metric


@pytest.mark.unit_tests
def test_ratio_above_threshold():
    """
    Test the ratio_above_threshold metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output and gt_ratio with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """

    # Test with default elevation threshold ----------------------------
    # Initialize input data
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)
    # Create metric object
    metric_obj = Metric("ratio_above_threshold")
    # Define elevation thresholds as default
    elevation_thrlds = metric_obj._ELEVATION_THRESHOLDS

    # Manually compute metric
    gt_ratio = []
    for threshold in elevation_thrlds:
        gt_ratio.append(
            ((np.count_nonzero(data > threshold)) / float(data.size))
        )
    gt_output = (gt_ratio, elevation_thrlds)
    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    np.testing.assert_equal(gt_output, output)

    # Test with custom elevation threshold ----------------------------

    elevation_thrlds = [-3, 2, 90]
    # Create metric object with custom parameter
    metric_obj = Metric(
        "ratio_above_threshold",
        params={"elevation_threshold": elevation_thrlds},
    )

    # Manually compute metric
    gt_ratio = []
    for threshold in elevation_thrlds:
        gt_ratio.append(
            ((np.count_nonzero(data > threshold)) / float(data.size))
        )
    gt_output = (gt_ratio, elevation_thrlds)

    # Compute metric from metric class
    output = metric_obj.compute_metric(data)

    np.testing.assert_equal(gt_output, output)


@pytest.mark.unit_tests
def test_cdf():
    """
    Test the cdf metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output, gt_bins_count and gt_cdf
    with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """

    # Initialize input data
    data = np.array(
        [
            [-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0],
            [-1.0, 3.0, 4.0, 5, 3.0, -6.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Create metric object
    metric_obj = Metric("cdf")
    # Manually compute metric
    # Get max diff from data
    max_diff = np.nanmax(np.abs(data))
    # Get bins number for histogram
    bin_step = metric_obj._BIN_STEP
    nb_bins = int(max_diff / bin_step)
    # getting data of the histogram
    hist, gt_bins_count = np.histogram(
        np.abs(data),
        bins=nb_bins,
        range=(0, max_diff),
        density=True,
    )
    # Normalized Probability Density Function of the histogram
    pdf = hist / sum(hist)
    # Generate Cumulative Probability Function
    gt_cdf = np.cumsum(pdf)

    # Compute metric from metric class
    output_cdf, output_bins_count = metric_obj.compute_metric(data)

    np.testing.assert_equal(output_cdf, gt_cdf)
    np.testing.assert_equal(output_bins_count, gt_bins_count)


@pytest.mark.unit_tests
def test_pdf():
    """
    Test the pdf metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output, gt_bins and gt_pdf with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    # Test without percentil 98 filtering ---------------------------

    # Initialize input data
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)

    # Create metric object
    metric_obj = Metric("pdf", params={"filter_p98": False})

    bin_step = metric_obj._BIN_STEP
    hist, gt_bins = np.histogram(
        data[~np.isnan(data)],
        bins=np.arange(-np.nanmax(data), np.nanmax(data), bin_step),
    )

    # Normalized Probability Density Function of the histogram
    gt_pdf = hist / sum(hist)

    # Compute metric from metric class
    output_pdf, output_bins = metric_obj.compute_metric(data)

    np.testing.assert_equal(output_pdf, gt_pdf)
    np.testing.assert_equal(output_bins, gt_bins)

    # Test with percentil 98 filtering ---------------------------

    # Initialize input data
    data = np.array([-7.0, 3.0, 3.0, 1, 3.0, 1.0, 0.0], dtype=np.float32)

    # Create metric object
    metric_obj = Metric("pdf", params={"filter_p98": True})

    bound = np.abs(np.nanpercentile(data, 98))
    bin_step = metric_obj._BIN_STEP
    hist, gt_bins = np.histogram(
        data[~np.isnan(data)],
        bins=np.arange(-bound, bound, bin_step),
    )

    # Normalized Probability Density Function of the histogram
    gt_pdf = hist / sum(hist)

    # Compute metric from metric class
    output_pdf, output_bins = metric_obj.compute_metric(data)

    np.testing.assert_equal(output_pdf, gt_pdf)
    np.testing.assert_equal(output_bins, gt_bins)


@pytest.mark.unit_tests
def test_slope_orientation_histogram():
    """
    Test the slope-orientation-histogram metric class function compute_metric.
    Input data:
    - Manually computed data array
    - Coregistration object created by fixture initialize_dem_and_coreg
    Validation data:
    - Manually computed ground truth gt_output, gt_bins and gt= with numpy
    Validation process:
    - Create the metric object and test compute_metric
    - Check that the obtained metrics are the same as ground truth
    """
    # Test without percentil 98 filtering ---------------------------

    # Initialize input data
    data = np.array(
        [
            [1982.0, 1967.0, 1950.0],
            [2005.0, 1988.0, 1969.0],
            [2012.0, 1990.0, 1967.0],
        ],
        dtype=np.float32,
    )

    # Create metric object
    metric_obj = Metric("slope-orientation-histogram")

    metric_obj.dx = -0.0008333333333333334
    metric_obj.dy = -0.0008333333333333334

    # Compute metric from metric class
    output, output_bins = metric_obj.compute_metric(data)

    gt = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    gt_bins = [
        0.5779019369622457,
        0.5886982642473321,
        0.5994945915324185,
        0.6102909188175049,
        0.6210872461025913,
        0.6318835733876776,
        0.642679900672764,
        0.6534762279578504,
        0.6642725552429368,
        0.6750688825280232,
        0.6858652098131096,
        0.696661537098196,
        0.7074578643832824,
        0.7182541916683688,
        0.7290505189534552,
        0.7398468462385416,
        0.750643173523628,
        0.7614395008087144,
        0.7722358280938008,
        0.7830321553788872,
        0.7938284826639735,
        0.8046248099490599,
        0.8154211372341463,
        0.8262174645192327,
        0.8370137918043191,
        0.8478101190894055,
        0.8586064463744919,
        0.8694027736595783,
        0.8801991009446647,
        0.8909954282297511,
        0.9017917555148375,
        0.9125880827999239,
        0.9233844100850103,
        0.9341807373700967,
        0.944977064655183,
        0.9557733919402694,
        0.9665697192253558,
        0.9773660465104422,
        0.9881623737955286,
        0.998958701080615,
        1.0097550283657015,
        1.0205513556507877,
        1.0313476829358743,
        1.0421440102209605,
        1.052940337506047,
        1.0637366647911333,
        1.0745329920762199,
        1.085329319361306,
        1.0961256466463927,
        1.1069219739314788,
        1.1177183012165655,
        1.1285146285016516,
        1.1393109557867382,
        1.1501072830718244,
        1.160903610356911,
        1.1716999376419972,
        1.1824962649270838,
        1.19329259221217,
        1.2040889194972566,
        1.2148852467823428,
        1.2256815740674294,
        1.2364779013525156,
        1.2472742286376022,
        1.2580705559226883,
        1.268866883207775,
        1.2796632104928611,
        1.2904595377779478,
        1.301255865063034,
        1.3120521923481205,
        1.3228485196332067,
        1.3336448469182933,
        1.3444411742033795,
        1.3552375014884661,
        1.3660338287735523,
        1.376830156058639,
        1.387626483343725,
        1.3984228106288117,
        1.4092191379138979,
        1.4200154651989845,
        1.4308117924840706,
        1.4416081197691573,
        1.4524044470542434,
        1.46320077433933,
        1.4739971016244162,
        1.4847934289095028,
        1.495589756194589,
        1.5063860834796756,
        1.5171824107647618,
        1.5279787380498484,
        1.5387750653349346,
        1.5495713926200212,
        1.5603677199051074,
        1.571164047190194,
        1.5819603744752802,
        1.5927567017603668,
        1.603553029045453,
        1.6143493563305396,
        1.6251456836156257,
        1.6359420109007123,
        1.6467383381857985,
        1.6575346654708851,
    ]

    np.testing.assert_equal(output, gt)
    np.testing.assert_equal(output_bins, gt_bins)
