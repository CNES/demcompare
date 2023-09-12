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
Mainly contains different matrice metric classes
"""
# pylint:disable=too-many-lines

from typing import Dict

import matplotlib.pyplot as mpl_pyplot
import numpy as np

# Third party imports
from numpy.fft import fft2, ifft2, ifftshift

from demcompare.img_tools import calc_spatial_freq_2d, neighbour_interpol

from .metric import Metric
from .metric_template import MetricTemplate


@Metric.register("dem-hill-shade")
class DemHillShade(MetricTemplate):
    """
    Compute the hill shade and optionnally save plots from a dem
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the matrice metric object
        :return: None
        """

        super().__init__()

        self.type = "matrice"
        self.fig_title = "REF dem hill shade"
        self.colorbar_title = "Hill shade"

        # angular direction of the sun
        self.azimuth: int = 315
        # angle of the illumination source above the horizon
        self.angle_altitude: int = 45
        self.cmap: str = "Greys_r"
        self.plot_path: str = None

        if parameters:
            if "azimuth" in parameters:
                self.azimuth = parameters["azimuth"]
            if "angle_altitude" in parameters:
                self.angle_altitude = parameters["angle_altitude"]
            if "cmap" in parameters:
                self.cmap = parameters["cmap"]
            if "colorbar_title" in parameters:
                self.colorbar_title = parameters["colorbar_title"]
            if "fig_title" in parameters:
                self.fig_title = parameters["fig_title"]
            if "plot_path" in parameters:
                self.plot_path = parameters["plot_path"]

    def compute_metric(self, data: np.ndarray) -> None:
        """
        Compute and optionnally save plots the hillshade view
        a of a dem using pyplot img_show.
        see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html # noqa: E501, B950 # pylint: disable=line-too-long

        :param data: input data to compute the metric
        :type data: np.array
        :return: None
        """

        x, y = np.gradient(data)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        azimuthrad = self.azimuth * np.pi / 180.0
        altituderad = self.angle_altitude * np.pi / 180.0

        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(
            altituderad
        ) * np.cos(slope) * np.cos(azimuthrad - aspect)

        hillshade_array = 255 * (shaded + 1) / 2

        fig, fig_ax = mpl_pyplot.subplots(figsize=(7.0, 8.0))

        image = mpl_pyplot.imshow(
            hillshade_array, cmap=mpl_pyplot.colormaps.get_cmap(self.cmap)
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


@Metric.register("dem-sky-view-factor")
class DemSkyViewFactor(MetricTemplate):
    """
    Compute the sky vuew factor and optionnally save plots from a dem
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the matrice metric object
        :return: None
        """

        super().__init__()

        self.type = "matrice"
        self.fig_title = "REF dem sky view factor"
        self.colorbar_title = "Sky view factor"

        # parameter of the DEM's FFT filter intensity.
        # Should be close to 1.
        self.filter_intensity: float = 0.9
        # if true, the image is replicated by x4 in order to improve resolution. # noqa: E501, B950 # pylint: disable=line-too-long
        self.replication: bool = True
        # quantiles
        self.quantiles = [0.09, 0.91]
        self.cmap: str = "Greys_r"
        self.plot_path: str = None

        if parameters:
            if "filter_intensity" in parameters:
                self.filter_intensity = parameters["filter_intensity"]
            if "replication" in parameters:
                self.replication = parameters["replication"]
            if "quantiles" in parameters:
                self.quantiles = parameters["quantiles"]
            if "cmap" in parameters:
                self.cmap = parameters["cmap"]
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
        Then, apply a filter y^filter_intensity with s=0.9: F(y) = F(y)* y^filter_intensity. # noqa: E501, B950 # pylint: disable=line-too-long
        Finally, apply the inverse FFT: IFFT(F(y)).
        We keep the real part (imaginary part = digital noise).

        :param data: input data to compute the metric
        :type data: np.array
        :return: curvature np.array containing :
        :rtype: np.array
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

    def compute_metric(self, data: np.ndarray) -> None:
        """
        Compute and optionnally save plots the sky view factor
        a of a dem using pyplot img_show.
        see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html # noqa: E501, B950 # pylint: disable=line-too-long

        :param data: input data to compute the metric
        :type data: np.array
        :return: None
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

        image = mpl_pyplot.imshow(
            z, cmap=mpl_pyplot.colormaps.get_cmap(self.cmap)
        )

        fig.colorbar(image, label=self.colorbar_title, ax=fig_ax)

        fig.text(
            0.15,
            0.15,
            f"Filter intensity = {self.filter_intensity}\nReplication={self.replication}\nQuantiles={self.quantiles}",  # noqa: E501, B950 # pylint: disable=line-too-long
            fontsize="medium",
        )

        if self.fig_title:
            fig_ax.set_title(self.fig_title, fontsize="large")

        if self.plot_path:
            mpl_pyplot.savefig(self.plot_path, dpi=100, bbox_inches="tight")

        mpl_pyplot.close()
