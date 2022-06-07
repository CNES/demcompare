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
This module contains functions associated to the
Nuth and Kaab universal co-registration
(Correcting elevation data for glacier change detection 2011).

Based on the work of geoutils project
https://github.com/GeoUtils/geoutils/blob/master/geoutils/dem_coregistration.py
Authors : Amaury Dehecq, Andrew Tedstone
Date : June 2015
License : MIT
"""

# Standard imports
import logging
import os
from typing import Tuple, Union

# Third party imports
import matplotlib.pyplot as pl
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import leastsq

# Demcompare imports
from ..dem_tools import create_dem
from ..initialization import ConfigType
from ..output_tree_design import get_out_dir
from ..transformation import Transformation
from .coregistration import Coregistration
from .coregistration_template import CoregistrationTemplate


@Coregistration.register("nuth_kaab_internal")
class NuthKaabInternal(CoregistrationTemplate):
    """
    NuthKaab class, allows to perform a Nuth & Kaab coregistration
    from authors above and adapted in demcompare
    """

    # Default parameters in case they are not specified in the cfg
    DEFAULT_ITERATIONS = 6
    # Method name
    method_name = "nuth_kaab_internal"

    def __init__(self, cfg: ConfigType = None):
        """
        Any coregistration class should have the following schema on
        its input cfg (optional parameters may be added for a
        particular coregistration class):

        coregistration = {
         "method_name": coregistration class name. str,
         "number_of_iterations": number of iterations. int,
         "sampling_source": optional. sampling source at which
           the dems are reprojected prior to coregistration. str
           "dem_to_align" (default) or "ref",
         "estimated_initial_shift_x": optional. estimated initial
           x shift. int or float. 0 by default,
         "estimated_initial_shift_y": optional. estimated initial
           y shift. int or float. 0 by default,
         "output_dir": optional output directory. str. If given,
           the coreg_dem is saved,
         "save_coreg_method_outputs": optional. bool. Requires output_dir
           to be set. If activated, the outputs of the coregistration method
           (such as nuth et kaab iteration plots) are saved,
         "save_internal_dems": optional. bool. Requires output_dir to be set.
           If activated, the internal dems of the coregistration
           such as reproj_dem, reproj_ref, reproj_coreg_dem,
           reproj_coreg_ref, initial_dh and final_dh are saved.
        }

        :param cfg: configuration
        :type cfg: ConfigType
        """
        # Call generic init before supercharging
        super().__init__(cfg)
        # Number of iterations specific to Nuth et kaab internal algorithm
        self.iterations = self.cfg["number_of_iterations"]
        # Aspect bounds for the Nuth et kaab internal algorithm
        self.aspect_bounds: np.array = None

    def fill_conf_and_schema(self, cfg: ConfigType = None) -> ConfigType:
        """
        Add default values to the dictionary if there are missing
        elements and define the configuration schema

        :param cfg: coregistration configuration
        :type cfg: ConfigType
        :return cfg: coregistration configuration updated
        :rtype: ConfigType
        """
        # Call generic fill_conf_and_schema
        cfg = super().fill_conf_and_schema(cfg)

        # Give the default value if the required element
        # is not in the configuration
        if "number_of_iterations" not in cfg:
            cfg["number_of_iterations"] = self.DEFAULT_ITERATIONS

        # Add subclass parameter to the default schema
        self.schema["number_of_iterations"] = cfg["number_of_iterations"]

        return cfg

    def _coregister_dems_algorithm(  # pylint:disable=too-many-locals
        self,
        dem_to_align: xr.Dataset,
        ref: xr.Dataset,
    ) -> Tuple[Transformation, xr.Dataset, xr.Dataset]:
        """
        Coregister_dems_algorithm, computes coregistration
        transformation and reprojected coregistered DEMs
        with Nuth et kaab algorithm
        Plots might be saved if save_coreg_method_outputs is set.

        :param dem_to_align: dem_to_align xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :type dem_to_align: xarray Dataset
        :param ref: ref xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :type ref: xarray Dataset
        :return: transformation, reproj_coreg_dem_to_align xr.DataSet,
                 reproj_coreg_ref xr.DataSet. The xr.Datasets containing :

                 - image : 2D (row, col) xr.DataArray float32
                 - georef_transform: 1D (trans_len) xr.DataArray
        :rtype: Tuple[Transformation, xr.Dataset, xr.Dataset]
        """
        # Copy dataset and extract image array
        dem_im = dem_to_align["image"].data
        ref_im = ref["image"].data

        # Set target dem grid for interpolation purpose
        xgrid = np.arange(dem_im.shape[1])
        ygrid = np.arange(dem_im.shape[0])

        # Compute inital_dh and initialize coreg_ref
        initial_dh = ref_im - dem_im
        coreg_ref = ref_im

        # Compute median, nmad and initial elevation difference plot
        median = np.median(initial_dh[np.isfinite(initial_dh)])
        nmad_old = 1.4826 * np.median(
            np.abs(initial_dh[np.isfinite(initial_dh)] - median)
        )
        maxval = 3 * nmad_old
        pl.figure(1, figsize=(7.0, 8.0))
        pl.imshow(initial_dh, vmin=-maxval, vmax=maxval)
        color_bar = pl.colorbar()
        color_bar.set_label("Elevation difference (m)")
        if self.save_coreg_method_outputs:
            output_dir_ = os.path.join(
                self.output_dir, get_out_dir("nuth_kaab_tmp_dir")
            )
            pl.savefig(
                os.path.join(output_dir_, "ElevationDiff_BeforeCoreg.png"),
                dpi=100,
                bbox_inches="tight",
            )
        pl.close()

        # Since later interpolations will consider
        # nodata values as normal values,
        # we need to keep track of nodata values
        # to get rid of them when the time comes
        nan_maskval = np.isnan(coreg_ref)
        dsm_from_filled = np.where(nan_maskval, -9999, coreg_ref)

        # Create spline function for interpolation
        spline_1 = RectBivariateSpline(
            ygrid, xgrid, dsm_from_filled, kx=1, ky=1
        )
        spline_2 = RectBivariateSpline(ygrid, xgrid, nan_maskval, kx=1, ky=1)
        x_offset, y_offset = 0, 0

        logging.info("Nuth & Kaab iterations: {}".format(self.iterations))
        coreg_dem = dem_im

        # Compute bounds for different aspect slices
        self.aspect_bounds = np.arange(0, 2 * np.pi, np.pi / 36)
        for i in range(self.iterations):
            # Remove bias from ref
            coreg_ref -= median
            # Compute new elevation difference
            dh = coreg_dem - coreg_ref
            # Compute slope and aspect
            slope, aspect = self._grad2d(coreg_dem)

            if self.save_coreg_method_outputs:
                output_dir_ = os.path.join(
                    self.output_dir, get_out_dir("nuth_kaab_tmp_dir")
                )
                plotfile = os.path.join(
                    output_dir_, "nuth_kaab_iter#{}.png".format(i)
                )
            else:
                plotfile = None

            # Compute offset
            east, north, z = self._nuth_kaab_single_iter(
                dh, slope, aspect, plot_file=plotfile
            )

            logging.info(
                "# {} - Offset in pixels : "
                "({:.2f},{:.2f}), -bias : ({:.2f})".format(
                    i + 1, east, north, z
                )
            )
            # Update total offsets
            x_offset += east
            y_offset += north

            # Resample slave DEM in the new grid
            # spline 1 : positive y shift moves south
            znew = spline_1(ygrid - y_offset, xgrid + x_offset)
            nanval_new = spline_2(ygrid - y_offset, xgrid + x_offset)

            # We created nan_maskval so that non NaN values are set to 0.
            # Interpolation "creates" values, and the values not affected
            # by nan are the ones still equal to 0.
            # Hence, all values different to 0 must be considered
            # as invalid ones
            znew[nanval_new != 0] = np.nan

            # Crop DEMs with offset
            if x_offset >= 0:
                coreg_ref = znew[:, 0 : znew.shape[1] - int(np.ceil(x_offset))]
                coreg_dem = dem_im[
                    :, 0 : dem_im.shape[1] - int(np.ceil(x_offset))
                ]
            else:
                coreg_ref = znew[:, int(np.floor(-x_offset)) : znew.shape[1]]
                coreg_dem = dem_im[
                    :, int(np.floor(-x_offset)) : dem_im.shape[1]
                ]
            if -y_offset >= 0:
                coreg_ref = coreg_ref[
                    0 : znew.shape[0] - int(np.ceil(-y_offset)), :
                ]
                coreg_dem = coreg_dem[
                    0 : dem_im.shape[0] - int(np.ceil(-y_offset)), :
                ]
            else:
                coreg_ref = coreg_ref[
                    int(np.floor(y_offset)) : znew.shape[0], :
                ]
                coreg_dem = coreg_dem[
                    int(np.floor(y_offset)) : dem_im.shape[0], :
                ]

            # Logging of some statistics
            diff = coreg_ref - coreg_dem
            diff = diff[np.isfinite(diff)]
            nmad_new = 1.4826 * np.median(np.abs(diff - np.median(diff)))
            median = np.median(diff)

            logging.info(
                (
                    "\tMedian : {0:.2f}, NMAD = {1:.2f}, Gain : {2:.2f}".format(
                        median, nmad_new, (nmad_new - nmad_old) / nmad_old * 100
                    )
                )
            )
            nmad_old = nmad_new

        # Generate the dataset dems
        coreg_dem_dataset = create_dem(
            coreg_dem,
            transform=dem_to_align.georef_transform.data,
            no_data=-32768,
            img_crs=dem_to_align.crs,
        )
        coreg_ref_dataset = create_dem(
            coreg_ref,
            transform=dem_to_align.georef_transform.data,
            no_data=-32768,
            img_crs=dem_to_align.crs,
        )
        logging.info(
            "Nuth & Kaab Final Offset in pixels (east, north):"
            "({:.2f},{:.2f})".format(x_offset, y_offset)
        )
        # Display
        final_dh = coreg_ref - coreg_dem
        median = np.median(final_dh[np.isfinite(final_dh)])
        nmad_old = 1.4826 * np.median(
            np.abs(final_dh[np.isfinite(final_dh)] - median)
        )
        maxval = 3 * nmad_old
        pl.figure(1, figsize=(7.0, 8.0))
        pl.imshow(final_dh, vmin=-maxval, vmax=maxval)
        color_bar = pl.colorbar()
        color_bar.set_label("Elevation difference (m)")
        if self.save_coreg_method_outputs:
            output_dir_ = os.path.join(
                self.output_dir, get_out_dir("nuth_kaab_tmp_dir")
            )
            pl.savefig(
                os.path.join(output_dir_, "ElevationDiff_AfterCoreg.png"),
                dpi=100,
                bbox_inches="tight",
            )
        z_offset = float(np.nanmean(final_dh))
        transform = Transformation(
            x_offset=x_offset,
            y_offset=-y_offset,  # -y_offset because y_offset
            # from nk is north oriented
            z_offset=z_offset,
            estimated_initial_shift_x=self.estimated_initial_shift_x,
            estimated_initial_shift_y=self.estimated_initial_shift_y,
            adapting_factor=self.adapting_factor,
        )
        return transform, coreg_dem_dataset, coreg_ref_dataset

    @staticmethod
    def _grad2d(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes input DEM's slope and aspect

        :param dem: input dem
        :type dem: np.ndarray
        :return: slope (fast forward style) and aspect
        :rtype: np.ndarray, np.ndarray
        """
        grad2, grad1 = np.gradient(dem)

        slope = np.sqrt(grad1**2 + grad2**2)
        aspect = np.arctan2(-grad1, grad2)  # aspect=0 when slope facing north
        aspect = aspect + np.pi

        return slope, aspect

    def _nuth_kaab_single_iter(
        self,
        dh: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
        plot_file: Union[str, bool] = None,
    ) -> Tuple[float, float, float]:
        """
        Computes the horizontal shift between 2 DEMs
        using the method presented in Nuth & Kaab 2011

        :param dh: elevation difference dem_to_align - ref
        :type dh: np.ndarray
        :param slope: slope for the same locations as the dh
        :type slope: np.ndarray
        :param aspect: aspect for the same locations as the dh
        :type aspect: np.ndarray
        :param plot_file: file to where store plot. Set to None if
            plot is to be printed. Set to False for no plot at all.
        :type plot_file: str or bool
        :return: east, north, c
        :rtype: float, float, float
        """
        # Compute estimated easting and northing of the shift,
        #         c is not used here but is related to the vertical shift
        # The aim is to compute dh / tan(alpha) as a function of the aspect
        # -> hence we are going to be slice the aspect to average a value
        #     for dh / tan(alpha) on those sliced areas
        # -> then we are going to fit the values by the model a.cos(b-aspect)+c
        #    - a will be the magnitude of the horizontal shift
        #    - b will be its orientation
        #    - c will be a vertical mean shift

        # To avoid nearly-zero division, filter slope values below 0.001
        slope[np.where(slope < 0.001)] = np.nan

        # function to be correlated with terrain aspect
        # NB : target = dh / tan(alpha) (see Fig. 2 of Nuth & Kaab 2011)
        # Explicitely ignore divide by zero warning,
        #   as they will be processed as nan later.
        with np.errstate(divide="ignore", invalid="ignore"):
            target = dh / slope
        target = target[np.isfinite(dh)]
        aspect = aspect[np.isfinite(dh)]

        # Compute filtered target
        slice_filt_median, target_filt = self._filter_target(aspect, target)

        # function to fit according to Nuth & Kaab
        x = aspect.ravel()
        y = target_filt.ravel()
        # remove non-finite values
        yf = y[(np.isfinite(x)) & (np.isfinite(y))]

        # set the first guess
        p0 = (3 * np.std(yf) / (2**0.5), 0, np.mean(yf))

        # least square fit
        def peval(x, p):
            """peval defines the model chosen"""
            return p[0] * np.cos(p[1] - x) + p[2]

        def residuals(p, y, x):
            """residuals function based on peval"""
            err = peval(x, p) - y
            return err

        # we run the least square fit
        # by minimizing the "distance" between y and peval (see residuals())
        plsq = leastsq(
            residuals,
            p0,
            args=(slice_filt_median, self.aspect_bounds),
            full_output=1,
        )
        yfit = peval(self.aspect_bounds, plsq[0])
        if plot_file:
            self._save_fit_plots(
                plot_file, aspect, target, target_filt, slice_filt_median, yfit
            )
        a, b, c = plsq[0]
        east = a * np.sin(b)  # with b=0 when north (origin=y-axis)
        north = a * np.cos(b)

        return east, north, c

    def _filter_target(
        self, aspect: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter target slice outliers of an input
        array by a 3*sigma filtering to improve Nuth et kaab fit.

        :param aspect: elevation difference dem_to_align - ref
        :type aspect: np.ndarray
        :param target: slope for the same locations as the dh
        :type target: np.ndarray
        :return: slice_filt_median, target_filt
        :rtype: Tuple[List]
        """
        # Define sigma to filter each target slice outliers
        # and improve Nuth et kaab fit.
        # All target values slice outside
        # [mean_slice-sigma_filter*std_slice, mean_slice+sigma_filter*std_slice]
        # are considered outliers and will be set to NaN
        sigma_filter = 3
        # Initialize slice filtered median
        slice_filt_median = []
        # Initialize filtered target
        target_filt = np.full(target.shape, np.nan)
        for bounds in self.aspect_bounds:
            # Slice indexes within aspect
            slice_idxes = np.where(
                (bounds < aspect) & (aspect < bounds + np.pi / 36)
            )
            # If no aspect values are within the slice,
            # fill mean with Nan and continue
            if len(slice_idxes[0]) == 0:
                # Set slice filtered median for Nuth et kaab as NaN
                slice_filt_median.append(np.nan)
                continue
            # Obtain target values in the slice
            target_slice = target[slice_idxes]
            # Obtain target slice's mean and std before filtering
            slice_mean = np.nanmean(target_slice)
            # numpy's std cannot handle nan
            slice_sigma = np.std(target_slice[np.isfinite(target_slice)])
            # Target slice values outside
            # [mean_slice-sigma_filter*std_slice,
            # mean_slice+sigma_filter*std_slice]
            # are considered outliers and set to NaN
            inv_idx = np.logical_or(
                (target_slice < (slice_mean - sigma_filter * slice_sigma)),
                (target_slice > (slice_mean + sigma_filter * slice_sigma)),
            )
            # Filter target_slice
            target_slice[inv_idx] = np.nan
            target_slice[inv_idx] = np.nan
            # Filter target
            target_filt[slice_idxes] = target_slice
            # Compute slice filtered median for Nuth et kaab
            slice_filt_median.append(np.nanmedian(target_slice))
        return np.array(slice_filt_median), np.array(target_filt)

    def _save_fit_plots(
        self,
        plot_file: str,
        aspect: np.ndarray,
        target: np.ndarray,
        target_filt: np.ndarray,
        slice_filt_median: np.ndarray,
        yfit: np.ndarray,
    ) -> None:
        """
        Compute and save the Nuth et Kaab fit plots of each iteration

        :param plot_file: file to where store plot
        :type plot_file: str
        :param aspect: terrain aspect
        :type aspect: np.ndarray
        :param target: nuth et kaab target
        :type target: np.ndarray
        :param target_filt: filtered target
        :type target_filt: np.ndarray
        :param slice_filt_median: filtered target median
        :type slice_filt_median: np.ndarray
        :param yfit: fit polynome of target_filt
        :type yfit: np.ndarray
        :return: None
        """

        # plotting results
        pl.figure(1, figsize=(7.0, 8.0))
        pl.plot(
            aspect * 180 / np.pi,
            target,
            ".",
            color="silver",
            markersize=3,
            label="target",
        )
        pl.plot(
            aspect * 180 / np.pi,
            target_filt,
            "c.",
            markersize=3,
            label="target filtered",
        )
        pl.plot(
            self.aspect_bounds * 180 / np.pi,
            slice_filt_median,
            "k.",
            label="median",
        )
        pl.plot(
            self.aspect_bounds * 180 / np.pi, yfit, "b-", label="median fit"
        )
        pl.xlabel("Terrain aspect (deg)")
        pl.ylabel(r"dh/tan($\alpha$) (meters)")
        # set axes limit on twice the min/max of the median,
        # or +-2 if the value is below it
        axes = pl.gca()
        ax_min = np.min([np.nanmin(slice_filt_median) * 2, -2])
        ax_max = np.max([np.nanmax(slice_filt_median) * 2, 2])
        axes.set_ylim([ax_min, ax_max])
        pl.legend(loc="upper left")
        pl.savefig(plot_file, dpi=100, bbox_inches="tight")
        pl.close()

    def compute_results(self):
        """
        Save the coregistration results on a Dict
        The altimetric and coregistration results are saved.
        Logging of the altimetric results is done in this function.

        :return: None
        """
        # Call generic compute_results before supercharging
        super().compute_results()
        # Add Nuth offsets to demcompare_results
        self.demcompare_results["coregistration_results"]["dx"][
            "nuth_offset"
        ] = round(self.transform.x_offset, 5)
        self.demcompare_results["coregistration_results"]["dy"][
            "nuth_offset"
        ] = round(self.transform.y_offset, 5)
