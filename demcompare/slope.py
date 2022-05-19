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
Slope computation from an Xarray dataset
TODO: move with stats refacto
"""

# Standards imports
from typing import Union

# Third party imports
import numpy as np
import xarray as xr
from scipy.ndimage import convolve

# Demcompare imports
from demcompare.img_tools import convert_pix_to_coord


def get_slope(dataset: xr.Dataset, degree: bool = False) -> np.ndarray:
    """
    Computes DEM's slope
    Slope is presented here :
    http://pro.arcgis.com/ \
            fr/pro-app/tool-reference/spatial-analyst/how-aspect-works.htm

    :param dataset: dataset
    :type dataset: xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
    :param degree:  True if is in degree
    :type degree: bool
    :return: slope
    :rtype: np.ndarray
    """

    # TODO: see if this function is moved with the refacto stats
    def get_orthodromic_distance(
        lon1: Union[float, np.ndarray],
        lat1: Union[float, np.ndarray],
        lon2: Union[float, np.ndarray],
        lat2: Union[float, np.ndarray],
    ):
        """
        Get Orthodromic distance from two (lat,lon) coordinates

        :param lon1: longitude 1
        :type lon1: Union[float, np.ndarray]
        :param lat1: latitude 1
        :type lat1: Union[float, np.ndarray]
        :param lon2: longitude 2
        :type lon2: Union[float, np.ndarray]
        :param lat2: latitude 2
        :type lat2: Union[float, np.ndarray]
        :return: orthodromic distance
        """
        # WGS-84 equatorial radius in km
        radius_equator = 6378137.0
        return radius_equator * np.arccos(
            np.cos(lat1 * np.pi / 180)
            * np.cos(lat2 * np.pi / 180)
            * np.cos((lon2 - lon1) * np.pi / 180)
            + np.sin(lat1 * np.pi / 180) * np.sin(lat2 * np.pi / 180)
        )

    crs = dataset.attrs["crs"]
    if not crs.is_projected:
        # Our dem is not projected, we can't simply use the pixel resolution
        # -> we need to compute resolution between each point
        ny, nx = dataset["image"].data.shape
        xp = np.arange(nx)
        yp = np.arange(ny)
        xp, yp = np.meshgrid(xp, yp)
        lon, lat = convert_pix_to_coord(
            dataset["georef_transform"].data, yp, xp
        )
        lonr = np.roll(lon, 1, 1)
        latl = np.roll(lat, 1, 0)

        distx = get_orthodromic_distance(lon, lat, lonr, lat)
        disty = get_orthodromic_distance(lon, lat, lon, latl)

        # deal withs ingularities at edges
        distx[:, 0] = distx[:, 1]
        disty[0] = disty[1]
    else:
        distx = np.abs(dataset.attrs["xres"])
        disty = np.abs(dataset.attrs["yres"])

    conv_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    conv_y = conv_x.transpose()

    # Now we do the convolutions :
    gx = convolve(dataset["image"].data, conv_x, mode="reflect")
    gy = convolve(dataset["image"].data, conv_y, mode="reflect")

    # And eventually we do compute tan(slope) and aspect
    tan_slope = np.sqrt((gx / distx) ** 2 + (gy / disty) ** 2) / 8
    slope = np.arctan(tan_slope)

    # Just simple unit change as required
    if degree is False:
        slope *= 100
    else:
        slope = (slope * 180) / np.pi

    return slope
