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
Mainly contains different DEM representation classes
"""
# pylint:disable=too-many-lines

from typing import Dict

import matplotlib.pyplot as mpl_pyplot
import numpy as np

# Third party imports
import xarray as xr
from matplotlib.colors import ListedColormap
from numpy.fft import fft2, ifft2, ifftshift

from demcompare.img_tools import calc_spatial_freq_2d, neighbour_interpol

from .dem_representation import DemRepresentation
from .dem_representation_template import DemRepresentationTemplate


@DemRepresentation.register("dem")
class Dem(DemRepresentationTemplate):
    """
    Compute and optionnally save plots from a dem
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the DEM representation object
        :return: None
        """

        super().__init__()

        self.fig_title = "REF dem"
        self.colorbar_title = "Elevation (m)"

    def compute_and_save_image_plots(
        self,
        dem: xr.Dataset,
        plot_path: str = None,
        fig_title: str = None,
        colorbar_title: str = None,
        cmap: str = "terrain",
    ):
        """
        Compute and optionnally save plots from a dem using pyplot img_show.
        see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html # noqa: E501, B950 # pylint: disable=line-too-long

        :param dem: dem object to compute and save image plots
        :type dem: str
        :param plot_path: path to save the plots if present
        :type plot_path: str
        :param fig_title: optional plot figure title
        :type fig_title: str
        :param title_colorbar: optional dem path to save the original tif file
        :type title_colorbar: str
        :param cmap: registered colormap name used to map scalar data to colors.
        :type cmap: str
        """

        # Create and save plot using the dem_plot function

        # Compute mean of dem image data
        mu = np.nanmean(dem["image"].data)
        # Compute std also
        sigma = np.nanstd(dem["image"].data)

        # Plot with matplotlib.pyplot
        fig, fig_ax = mpl_pyplot.subplots(figsize=(7.0, 8.0))
        # add fig title if present
        if fig_title:
            fig_ax.set_title(fig_title, fontsize="large")
        #
        im1 = fig_ax.imshow(
            dem["image"].data,
            cmap=cmap,
            vmin=mu - sigma,
            vmax=mu + sigma,
            interpolation="none",
            aspect="equal",
        )
        fig.colorbar(im1, label=colorbar_title, ax=fig_ax)
        fig.text(
            0.15,
            0.15,
            f"Values rescaled between"
            f"\n[mean-std, mean+std]=[{mu - sigma:.2f}, {mu + sigma:.2f}]",
            fontsize="medium",
        )
        # Save plot
        if plot_path:
            mpl_pyplot.savefig(plot_path, dpi=100, bbox_inches="tight")

        mpl_pyplot.close()


@DemRepresentation.register("dem-hill-shade")
class DemHillShade(DemRepresentationTemplate):
    """
    Compute the hill shade and optionnally save plots from a dem
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the DEM representation object
        :return: None
        """

        super().__init__()

        self.fig_title = "REF dem hill shade"
        self.colorbar_title = "Hill shade"

        # angular direction of the sun
        self.azimuth: int = 315
        # angle of the illumination source above the horizon
        self.angle_altitude: int = 45

        if parameters:
            if "azimuth" in parameters:
                self.azimuth = parameters["azimuth"]
            if "angle_altitude" in parameters:
                self.angle_altitude = parameters["angle_altitude"]

    def compute_and_save_image_plots(
        self,
        dem: xr.Dataset,
        plot_path: str = None,
        fig_title: str = None,
        colorbar_title: str = None,
        cmap: str = "Greys_r",
        cmap2: str = "royalblue",
    ) -> None:
        """
        Compute and optionnally save plots the hillshade view
        a of a dem using pyplot img_show.
        see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html # noqa: E501, B950 # pylint: disable=line-too-long

        :param dem: dem object to compute and save image plots
        :type dem: str
        :param plot_path: path to save the plots if present
        :type plot_path: str
        :param fig_title: optional plot figure title
        :type fig_title: str
        :param title_colorbar: optional dem path to save the original tif file
        :type title_colorbar: str
        :param cmap: registered colormap name used to map scalar data to colors
        :type cmap: str
        :param cmap2: registered colormap name used to map nodata values to colors # noqa: E501, B950 # pylint: disable=line-too-long
        :type cmap2: str
        :return: None
        """

        x, y = np.gradient(dem["image"].data)
        slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
        aspect = np.arctan2(-x, y)
        azimuthrad = self.azimuth * np.pi / 180.0
        altituderad = self.angle_altitude * np.pi / 180.0

        shaded = np.sin(altituderad) * np.sin(slope) + np.cos(
            altituderad
        ) * np.cos(slope) * np.cos(azimuthrad - aspect)

        hillshade_array = 255 * (shaded + 1) / 2

        fig, fig_ax = mpl_pyplot.subplots(figsize=(7.0, 8.0))

        no_data_location = np.logical_or(
            dem["image"].data == dem.attrs["nodata"],
            np.isnan(dem["image"].data),
        )

        mpl_pyplot.imshow(no_data_location, cmap=ListedColormap(cmap2))
        im2 = mpl_pyplot.imshow(
            hillshade_array, cmap=mpl_pyplot.colormaps.get_cmap(cmap)
        )

        fig.colorbar(im2, label=colorbar_title, ax=fig_ax)

        fig.text(
            0.15,
            0.15,
            f"Azimuth={self.azimuth}\nAngle altitude={self.angle_altitude}",
            fontsize="medium",
        )

        if fig_title:
            fig_ax.set_title(fig_title, fontsize="large")

        if plot_path:
            mpl_pyplot.savefig(plot_path, dpi=100, bbox_inches="tight")

        mpl_pyplot.close()


@DemRepresentation.register("dem-sky-view-factor")
class DemSkyViewFactor(DemRepresentationTemplate):
    """
    Compute the sky vuew factor and optionnally save plots from a dem
    """

    def __init__(self, parameters: Dict = None):
        """
        Initialization the DEM representation object
        :return: None
        """

        super().__init__()

        self.fig_title = "REF dem sky view factor"
        self.colorbar_title = "Sky view factor"

        # parameter of the DEM's FFT filter intensity.
        # Should be close to 1.
        self.filter_intensity: float = 0.9
        # if true, the image is replicated by x4 in order to improve resolution. # noqa: E501, B950 # pylint: disable=line-too-long
        self.replication: bool = True
        # quantiles
        self.quantiles = [0.09, 0.91]

        if parameters:
            if "filter_intensity" in parameters:
                self.filter_intensity = parameters["filter_intensity"]
            if "replication" in parameters:
                self.replication = parameters["replication"]
            if "quantiles" in parameters:
                self.quantiles = parameters["quantiles"]

    def compute_svf(
        self,
        dem: xr.Dataset,
    ):
        """
        Return the sky view factor of the input DEM.
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
        data_all = neighbour_interpol(dem["image"].data, no_data_location)

        high, wide = dem["image"].data.shape

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

    def compute_and_save_image_plots(
        self,
        dem: xr.Dataset,
        plot_path: str = None,
        fig_title: str = None,
        colorbar_title: str = None,
        cmap: str = "Greys_r",
        cmap2: str = "royalblue",
    ) -> None:
        """
        Compute and optionnally save plots the sky view factor
        a of a dem using pyplot img_show.
        see https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html # noqa: E501, B950 # pylint: disable=line-too-long

        :param dem: dem object to compute and save image plots
        :type dem: str
        :param plot_path: path to save the plots if present
        :type plot_path: str
        :param fig_title: optional plot figure title
        :type fig_title: str
        :param title_colorbar: optional dem path to save the original tif file
        :type title_colorbar: str
        :param cmap: registered colormap name used to map scalar data to colors
        :type cmap: str
        :param cmap2: registered colormap name used to map nodata values to colors
        :type cmap2: str
        :return: None
        """

        fig, fig_ax = mpl_pyplot.subplots(figsize=(7.0, 8.0))

        no_data_location = np.logical_or(
            dem["image"].data == dem.attrs["nodata"],
            np.isnan(dem["image"].data),
        )

        z = self.compute_svf(dem)

        z1d = z[~no_data_location].reshape(-1)

        a, b = np.quantile(
            z1d, [self.quantiles[0], self.quantiles[1]]
        )  # find thresholds that saturate 9%

        # rescale using a and b
        z = (z - a) / (b - a)

        # clip between 0 and 1 + rescale to 255
        z = np.clip(z, 0, 1) * 255

        # reset NaN values before plot
        z[no_data_location] = np.nan

        mpl_pyplot.imshow(no_data_location, cmap=ListedColormap(cmap2))
        im2 = mpl_pyplot.imshow(z, cmap=mpl_pyplot.colormaps.get_cmap(cmap))

        fig.colorbar(im2, label=colorbar_title, ax=fig_ax)

        fig.text(
            0.15,
            0.15,
            f"Filter intensity = {self.filter_intensity}\nReplication={self.replication}\nQuantiles={self.quantiles}",  # noqa: E501, B950 # pylint: disable=line-too-long
            fontsize="medium",
        )

        if fig_title:
            fig_ax.set_title(fig_title, fontsize="large")

        if plot_path:
            mpl_pyplot.savefig(plot_path, dpi=100, bbox_inches="tight")

        mpl_pyplot.close()
