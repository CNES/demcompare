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
Mainly contains different matrix metric classes
"""
from typing import Dict

import matplotlib.pyplot as mpl_pyplot
import numpy as np
import rasterio
import xarray as xr
from matplotlib.colors import ListedColormap

# Third party imports
from numpy.fft import fft2, ifft2, ifftshift

from demcompare.dem_tools import create_dem
from demcompare.img_tools import calc_spatial_freq_2d, neighbour_interpol

from .metric import Metric
from .metric_template import MetricTemplate


@Metric.register("hillshade")
class DemHillShade(MetricTemplate):
    """
    Compute the hill shade and optionnally save plots from a dem
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the matrix metric object
        :return: None
        """

        super().__init__()

        self.type = "matrix2D"
        self.fig_title = "DEM hill shade"
        self.colorbar_title = "Hill shade"

        # angular direction of the sun
        self.azimuth: float = 315
        # angle of the illumination source above the horizon
        self.angle_altitude: float = 45
        self.cmap: str = "Greys_r"
        self.cmap_nodata: str = "royalblue"
        self.plot_path: str = None
        self.no_data_location: np.ndarray = None
        self.bounds: rasterio.coords.BoundingBox = None

        if parameters:
            if "azimuth" in parameters:
                self.azimuth = parameters["azimuth"]
            if "angle_altitude" in parameters:
                self.angle_altitude = parameters["angle_altitude"]
            if "cmap" in parameters:
                self.cmap = parameters["cmap"]
            if "cmap_nodata" in parameters:
                self.cmap_nodata = parameters["cmap_nodata"]
            if "colorbar_title" in parameters:
                self.colorbar_title = parameters["colorbar_title"]
            if "fig_title" in parameters:
                self.fig_title = parameters["fig_title"]
            if "plot_path" in parameters:
                self.plot_path = parameters["plot_path"]

    def compute_hillshade(
        self, data: np.ndarray, azimuth: float, angle_altitude: float
    ) -> np.ndarray:
        """
        Compute the hillshade view a of a dem.

        :param data: input data to compute the metric
        :type data: np.array
        :param azimuth: angular direction of the sun
        :type azimuth: float
        :param angle_altitude: angle of the illumination source
         above the horizon
        :type angle_altitude: float
        :return: np.ndarray
        """

        x, y = np.gradient(data)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        azimuthrad = azimuth * np.pi / 180.0
        altituderad = angle_altitude * np.pi / 180.0

        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(
            altituderad
        ) * np.cos(slope) * np.cos(azimuthrad - aspect)

        hillshade_array = 255 * (shaded + 1) / 2

        return hillshade_array

    def compute_metric(self, data: np.ndarray) -> xr.Dataset:
        """
        Compute and optionnally save plots the hillshade view
        a of a dem using pyplot img_show.

        :param data: input data to compute the metric
        :type data: xr.Dataset
        :return: None
        """

        hillshade_array = self.compute_hillshade(
            data, self.azimuth, self.angle_altitude
        )

        fig, fig_ax = mpl_pyplot.subplots(figsize=(7.0, 8.0))

        no_data_location = self.no_data_location

        if no_data_location is not None:
            mpl_pyplot.imshow(
                no_data_location,
                cmap=ListedColormap([self.cmap_nodata]),
                interpolation="none",
                aspect="equal",
            )

            hillshade_array[no_data_location] = np.nan

        image = mpl_pyplot.imshow(
            hillshade_array,
            cmap=mpl_pyplot.colormaps.get_cmap(self.cmap),
        )

        fig.colorbar(image, label=self.colorbar_title, ax=fig_ax)

        fig.text(
            0.15,
            0.15,
            f"Azimuth={self.azimuth}\nAngle altitude={self.angle_altitude}",
            fontsize="medium",
        )

        if self.fig_title:
            fig_ax.set_title(self.fig_title, fontsize="large")

        if self.plot_path:
            mpl_pyplot.savefig(self.plot_path, dpi=100, bbox_inches="tight")

        mpl_pyplot.close()

        if self.bounds is not None:
            return create_dem(hillshade_array, bounds=self.bounds)
        return create_dem(hillshade_array)


@Metric.register("svf")
class DemSkyViewFactor(MetricTemplate):
    """
    Compute the sky vuew factor and optionnally save plots from a dem
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the matrix metric object
        :return: None
        """

        super().__init__()

        self.type = "matrix2D"
        self.fig_title = "DEM sky view factor"
        self.colorbar_title = "Sky view factor"

        # parameter of the DEM's FFT filter intensity.
        # Should be close to 1.
        self.filter_intensity: float = 0.9
        # if true, the image is replicated by x4
        # in order to improve resolution.
        self.replication: bool = True
        # quantiles
        self.quantiles = [0.09, 0.91]
        self.cmap: str = "Greys_r"
        self.cmap_nodata: str = "royalblue"
        self.plot_path: str = None
        self.no_data_location: np.ndarray = None
        self.bounds: rasterio.coords.BoundingBox = None

        if parameters:
            if "filter_intensity" in parameters:
                self.filter_intensity = parameters["filter_intensity"]
            if "replication" in parameters:
                self.replication = parameters["replication"]
            if "quantiles" in parameters:
                self.quantiles = parameters["quantiles"]
            if "cmap" in parameters:
                self.cmap = parameters["cmap"]
            if "cmap_nodata" in parameters:
                self.cmap_nodata = parameters["cmap_nodata"]
            if "colorbar_title" in parameters:
                self.colorbar_title = parameters["colorbar_title"]
            if "fig_title" in parameters:
                self.fig_title = parameters["fig_title"]
            if "plot_path" in parameters:
                self.plot_path = parameters["plot_path"]

    def compute_svf(
        self,
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Return the sky view factor of the input DEM.
        First, compute the FFT of the input dem: F(y) = FFT(DEM).
        Then, apply a filter y^filter_intensity
        with s=0.9: F(y) = F(y)* y^filter_intensity.
        Finally, apply the inverse FFT: IFFT(F(y)).
        We keep the real part (imaginary part = digital noise).

        :param data: input data to compute the metric
        :type data: np.array
        :return: curvature np.array containing :
        :rtype: np.ndarray
        """

        high, wide = data.shape

        no_data_location = np.isnan(data)

        # no data pixel interpolation
        data_all = neighbour_interpol(data, no_data_location)

        if self.replication:
            data_all = np.hstack([data_all, np.flip(data_all, axis=1)])
            data_all = np.vstack([data_all, np.flip(data_all, axis=0)])
            f_y, f_x = calc_spatial_freq_2d(2 * high, 2 * wide, edge=np.pi)

        else:
            f_y, f_x = calc_spatial_freq_2d(high, wide, edge=np.pi)

        # spatial frequency (module)
        res = (f_x**2 + f_y**2) ** (self.filter_intensity / 2)
        res = ifftshift(res)

        image = fft2(data_all)
        image_filtered = image * res

        image_filtered = ifft2(image_filtered)

        if self.replication:
            image_filtered = image_filtered[:high, :wide]

        # real part + thresholding to 0 (negative values are kept)
        image_filtered = np.fmin(0, image_filtered.real)

        return image_filtered

    def compute_metric(self, data: np.ndarray) -> xr.Dataset:
        """
        Compute and optionnally save plots the sky view factor
        a of a dem using pyplot img_show.

        :param data: input data to compute the metric
        :type data: np.ndarray
        :return: xr.Dataset
        """

        fig, fig_ax = mpl_pyplot.subplots(figsize=(7.0, 8.0))

        z = self.compute_svf(data)

        z1d = z.reshape(-1)

        a, b = np.quantile(
            z1d, [self.quantiles[0], self.quantiles[1]]
        )  # find thresholds that saturate 9%

        # rescale using a and b
        z = (z - a) / (b - a)

        # clip between 0 and 1 + rescale to 255
        z = np.clip(z, 0, 1) * 255

        no_data_location = self.no_data_location

        if no_data_location is not None:
            mpl_pyplot.imshow(
                no_data_location,
                cmap=ListedColormap([self.cmap_nodata]),
                interpolation="none",
                aspect="equal",
            )
            z[no_data_location] = np.nan

        image = mpl_pyplot.imshow(
            z, cmap=mpl_pyplot.colormaps.get_cmap(self.cmap)
        )

        fig.colorbar(image, label=self.colorbar_title, ax=fig_ax)

        fig.text(
            0.15,
            0.15,
            f"Filter intensity = {self.filter_intensity}"
            f"\nReplication={self.replication}"
            f"\nQuantiles={self.quantiles}",
            fontsize="medium",
        )

        if self.fig_title:
            fig_ax.set_title(self.fig_title, fontsize="large")

        if self.plot_path:
            mpl_pyplot.savefig(self.plot_path, dpi=100, bbox_inches="tight")

        mpl_pyplot.close()

        if self.bounds is not None:
            return create_dem(z, bounds=self.bounds)
        return create_dem(z)
