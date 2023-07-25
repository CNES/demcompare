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
# pylint:disable=too-few-public-methods
"""
Mainly contains different DEM processing classes
"""

import logging
from typing import Dict

import numpy as np

# Third party imports
import xarray as xr
from numpy.fft import fft2, ifft2, ifftshift
from scipy.interpolate import griddata

from demcompare.dem_tools import accumulates_class_layers, create_dem

from .dem_processing import DemProcessing
from .dem_processing_template import DemProcessingTemplate


@DemProcessing.register("alti-diff")
class AltiDiff(DemProcessingTemplate):
    """
    Altitude difference between two DEMs
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the DEM processing object
        :return: None
        """

        super().__init__()

        self.fig_title = "[REF - SEC] difference"
        self.colorbar_title = "Elevation difference (m)"

    def compute_dems_diff(
        self,
        dem_1: xr.Dataset,
        dem_2: xr.Dataset,
    ) -> xr.Dataset:
        """
        Compute altitude difference dem_1 - dem_2 and
        return it as an xr.Dataset with the dem_2
        georeferencing and attributes.

        :param dem_1: dem_1 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem_1: xr.Dataset
         :param dem_2: dem_2 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem_2: xr.Dataset
        :return: difference xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :rtype: xr.Dataset
        """
        diff_raster = dem_1["image"].data - dem_2["image"].data

        diff_dem = create_dem(
            diff_raster,
            transform=dem_2.georef_transform.data,
            nodata=dem_1.attrs["nodata"],
            img_crs=dem_2.crs,
            bounds=dem_2.bounds,
        )
        return diff_dem

    def process_dem(
        self,
        dem_1: xr.Dataset,
        dem_2: xr.Dataset = None,
    ) -> xr.Dataset:
        """
        Compute the difference between dem_1 and dem_2.
        Add classification layers to the difference.

        :param dem_1: dem_1 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem_1: xr.Dataset
        :param dem_2: dem_2 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem_2: xr.Dataset
        :return: difference xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :rtype: xr.Dataset
        """
        diff = self.compute_dems_diff(dem_1, dem_2)
        diff = accumulates_class_layers(dem_1, dem_2, diff)
        return diff


@DemProcessing.register("ref")
class Ref(DemProcessingTemplate):
    """
    REF DEM
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the DEM processing object
        :return: None
        """

        super().__init__()

        self.fig_title = "REF dem"
        self.colorbar_title = "Elevation (m)"

    def process_dem(
        self,
        dem_1: xr.Dataset,
        dem_2: xr.Dataset = None,
    ) -> xr.Dataset:
        """
        Return dem_1

        :param dem_1: dem_1 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem_1: xr.Dataset
        :param dem_2: dem_2 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem_2: xr.Dataset
        :return: xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :rtype: xr.Dataset
        """

        if dem_2 is not None:
            logging.error(
                "The DEM processing method: %s,"
                " takes only one input to the process_dem function",
                self.type,
            )
            raise ValueError

        return dem_1


@DemProcessing.register("ref-curvature")
class RefCurvature(DemProcessingTemplate):
    """
    Curvature of the REF DEM
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the DEM processing object
        :return: None
        """

        super().__init__()

        self.fig_title = "REF dem curvature"
        self.colorbar_title = "Curvature"

    def compute_curvature_filtering(
        self,
        dem: xr.Dataset,
        filter_intensity: float = 0.9,
        replication: bool = True,
    ) -> xr.Dataset:
        """
        Return the curvature of the input dem.
        First, compute the FFT of the input dem: F(y) = FFT(DEM).
        Then, apply a filter y^filter_intensity with s=0.9: F(y) = F(y)* y^filter_intensity. # noqa: E501, B950 # pylint: disable=line-too-long
        Finally, apply the inverse FFT: IFFT(F(y)).
        We keep the real part (imaginary part = digital noise).

        :param dem: dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem: xr.Dataset
        :param filter_intensity: parameter of the DEM's FFT filter intensity.
                                 Should be close to 1.
                                 Default = 0.9.
        :type filter_intensity: float
        :param replication: if true, the image is replicated by x4 in order to improve resolution. # noqa: E501, B950 # pylint: disable=line-too-long
                            Default = True.
        :type replication: bool
        :return: curvature xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :rtype: xr.Dataset
        """

        no_data_location = np.logical_or(
            dem["image"].data == dem.attrs["nodata"],
            np.isnan(dem["image"].data),
        )

        # no data pixel interpolation
        data_all = self.neighbour_interpol(dem["image"].data, no_data_location)

        high, wide = dem["image"].data.shape

        if replication:
            data_all = np.hstack([data_all, np.flip(data_all, axis=1)])
            data_all = np.vstack([data_all, np.flip(data_all, axis=0)])
            f_y, f_x = self.calc_spatial_freq_2d(2 * high, 2 * wide, edge=np.pi)

        else:
            f_y, f_x = self.calc_spatial_freq_2d(high, wide, edge=np.pi)

        # spatial frequency (module)
        spatial_freq = (f_x**2 + f_y**2) ** (filter_intensity / 2)
        spatial_freq = ifftshift(spatial_freq)

        image = fft2(data_all)
        image_filtered = image * spatial_freq

        image_filtered = ifft2(image_filtered)

        if replication:
            image_filtered = image_filtered[:high, :wide]

        image_filtered[no_data_location] = dem.attrs["nodata"]

        return create_dem(
            image_filtered.real,
            transform=dem.georef_transform.data,
            nodata=dem.attrs["nodata"],
            img_crs=dem.crs,
            bounds=dem.bounds,
            classification_layer_masks=dem["classification_layer_masks"]
            if hasattr(dem, "classification_layer_masks")
            else None,
        )

    def neighbour_interpol(
        self, data2d: np.ndarray, no_values_location: np.ndarray
    ) -> np.ndarray:
        """
        Nearest neighbor interpolation function.
        Applied to DEM containing no data values for calculating curvature.

        :param data2d: 2D (row, col) np.ndarray containing the image
        :type data2d: np.ndarray
        :param no_values_location: 2D (row, col) np.ndarray
                                   containing the no data values
        :type no_values_location: np.ndarray
        :return: 2D interpolated np.ndarray
        :rtype: np.ndarray
        """

        if np.any(no_values_location):
            size_y, size_x = data2d.shape

            values_location = np.logical_not(no_values_location)

            x, y = np.meshgrid(range(size_x), range(size_y))

            data_holes = griddata(
                (y[values_location][:], x[values_location][:]),
                data2d[values_location][:],
                (y[no_values_location], x[no_values_location]),
                method="nearest",
            )

            data2d[
                y[no_values_location][:], x[no_values_location][:]
            ] = data_holes[:]

        return data2d

    def calc_spatial_freq_2d(self, s_y: int, s_x: int, edge: float = np.pi):
        """
        Calculation of normalized spatial frequencies
        for a 2D image/matrix of size s_y * s_x.

        :param s_y: number of rows of the image
        :type s_y: int
        :param s_x: number of columns of the image
        :type s_x: int
        :return: normalized spatial frequencies
        :rtype: (np.ndarray, np.ndarray)
        """

        f_x = self.calc_spatial_freq_1d(s_x, edge)
        f_y = self.calc_spatial_freq_1d(s_y, edge)

        f_x, f_y = np.meshgrid(f_x, f_y)

        return f_y, f_x

    def calc_spatial_freq_1d(self, n: int, edge: float = np.pi) -> np.ndarray:
        """
        Calculation of normalized frequencies
        between [-edge, +edge] of a vector of n samples.
        n is even or odd.
        In both cases, the vector contains the zero frequency at the center.

        :param n: frequency vector size
        :type n: int
        :param edge: maximum frequency
        :type edge: float
        :return: frequencies between -edge and +edge
        :rtype: np.ndarray
        """

        # if size is even: f_x is defined on [-edge, +edge[.
        if np.mod(n, 2) == 0:
            freq_neg = -np.arange(1, n / 2 + 1)[::-1]
            freq_pos = +np.arange(0, n / 2)

        # if size is odd: f_x is defined on [-edge, +edge].
        else:
            # use Euclidean division //
            freq_neg = -np.arange(1, n // 2 + 1)[::-1]  # type: ignore
            freq_pos = +np.arange(0, n // 2 + 1)  # type: ignore

        return np.concatenate((freq_neg, freq_pos)) * 2 * edge / n

    def process_dem(
        self,
        dem_1: xr.Dataset,
        dem_2: xr.Dataset = None,
    ) -> xr.Dataset:
        """
        Return the curvature of dem_1

        :param dem_1: dem_1 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem_1: xr.Dataset
        :param dem_2: dem_2 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem_2: xr.Dataset
        :return: curvature xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :rtype: xr.Dataset
        """

        if dem_2 is not None:
            logging.error(
                "The DEM processing method: %s,"
                " takes only one input to the process_dem function",
                self.type,
            )
            raise ValueError

        return self.compute_curvature_filtering(dem_1)
