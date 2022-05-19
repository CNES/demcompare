#!/usr/bin/env python
# coding: utf8
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
Nuth et Kaab coregistration method.
"""
import logging
import os
import sys
from typing import Dict, Tuple, Union

import matplotlib.pyplot as pl
import numpy as np
import xarray as xr
from json_checker import And, Checker, Or
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import leastsq

from ..dem_tools import SamplingSourceParameter, create_dem
from ..img_tools import compute_gdal_translate_bounds
from ..output_tree_design import get_out_dir
from ..transformation import Transformation
from .coregistration import Coregistration


class NuthKaab(Coregistration, method_name="nuth_kaab"):
    """
    NuthKaab class, allows to perform the coregistration
    """

    _SAMPLING_SOURCE = SamplingSourceParameter.DEM_TO_ALIGN.value
    _ITERATIONS = 6

    def __init__(self, **cfg: dict):
        """
        :param cfg: configuration
        :type cfg: dict
        """
        # Configuration file
        self.cfg = self._check_conf(**cfg)
        # Output directory to save results
        self.output_dir = self.cfg["output_dir"]
        # Number of iterations
        self.iterations = self.cfg["number_of_iterations"]
        # Sampling source considered during reprojection
        # (see dem_tools.SamplingSourceParameter)
        self.sampling_source = self.cfg["sampling_source"]
        # Estimated initial shift x
        self.estimated_initial_shift_x = self.cfg["estimated_initial_shift_x"]
        # Estimated initial shif y
        self.estimated_initial_shift_y = self.cfg["estimated_initial_shift_y"]
        # Save internal dems
        self.save_internal_dems = self.cfg["save_internal_dems"]
        # Save coreg_method outputs
        self.save_coreg_method_outputs = self.cfg["save_coreg_method_outputs"]
        # Aspect bounds for the Nuth et kaab algorithm
        self.aspect_bounds: np.array = None

    def _check_conf(
        self, **cfg: Union[str, float, int, SamplingSourceParameter]
    ) -> Dict[str, Union[str, float, int]]:
        """
        Add default values to the dictionary if there are missing
        elements and check if the dictionary is correct

        :param cfg: coregistration configuration
        :type cfg: dict
        :return cfg: coregistration configuration updated
        :rtype: dict
        """
        # Give the default value if the required element
        # is not in the configuration
        if "number_of_iterations" not in cfg:
            cfg["number_of_iterations"] = self._ITERATIONS
        if "sampling_source" not in cfg:
            cfg["sampling_source"] = self._SAMPLING_SOURCE
        if "estimated_initial_shift_x" not in cfg:
            cfg["estimated_initial_shift_x"] = 0
            cfg["estimated_initial_shift_y"] = 0
        if "save_internal_dems" in cfg:
            cfg["save_internal_dems"] = bool(cfg["save_internal_dems"])
        else:
            cfg["save_internal_dems"] = False
        if "save_coreg_method_outputs" in cfg:
            cfg["save_coreg_method_outputs"] = bool(
                cfg["save_coreg_method_outputs"]
            )
        else:
            cfg["save_coreg_method_outputs"] = False

        if "output_dir" not in cfg:
            cfg["output_dir"] = None
            if cfg["save_internal_dems"] or cfg["save_coreg_method_outputs"]:
                logging.error(
                    "save_internal_dems and/or save_coreg_method_outputs"
                    " options are activated but no output_dir has been set. "
                    "Please set the output_dir parameter or deactivate"
                    " the saving options."
                )
                sys.exit(1)

        schema = {
            "number_of_iterations": And(int, lambda x: x > 1),
            "sampling_source": And(
                str,
                Or(
                    lambda input: SamplingSourceParameter.REF.value,
                    lambda input: SamplingSourceParameter.DEM_TO_ALIGN.value,
                ),
            ),
            "estimated_initial_shift_x": Or(int, float),
            "estimated_initial_shift_y": Or(int, float),
            "method_name": And(str, lambda input: "nuth_kaab"),
            "output_dir": Or(str, None),
            "save_coreg_method_outputs": bool,
            "save_internal_dems": bool,
        }

        checker = Checker(schema)
        checker.validate(cfg)
        return cfg

    def _coregister_dems(  # pylint:disable=too-many-locals
        self,
        dem_to_align: xr.Dataset,
        ref: xr.Dataset,
    ) -> Tuple[Transformation, xr.Dataset, xr.Dataset]:
        """
        Coregister_dems, computes coregistration
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
        xoff, yoff = 0, 0

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
            xoff += east
            yoff += north

            # Resample slave DEM in the new grid
            # spline 1 : positive y shift moves south
            znew = spline_1(ygrid - yoff, xgrid + xoff)
            nanval_new = spline_2(ygrid - yoff, xgrid + xoff)

            # We created nan_maskval so that non NaN values are set to 0.
            # Interpolation "creates" values, and the values not affected
            # by nan are the ones still equal to 0.
            # Hence, all values different to 0 must be considered
            # as invalid ones
            znew[nanval_new != 0] = np.nan

            # Crop DEMs with offset
            if xoff >= 0:
                coreg_ref = znew[:, 0 : znew.shape[1] - int(np.ceil(xoff))]
                coreg_dem = dem_im[:, 0 : dem_im.shape[1] - int(np.ceil(xoff))]
            else:
                coreg_ref = znew[:, int(np.floor(-xoff)) : znew.shape[1]]
                coreg_dem = dem_im[:, int(np.floor(-xoff)) : dem_im.shape[1]]
            if -yoff >= 0:
                coreg_ref = coreg_ref[
                    0 : znew.shape[0] - int(np.ceil(-yoff)), :
                ]
                coreg_dem = coreg_dem[
                    0 : dem_im.shape[0] - int(np.ceil(-yoff)), :
                ]
            else:
                coreg_ref = coreg_ref[int(np.floor(yoff)) : znew.shape[0], :]
                coreg_dem = coreg_dem[int(np.floor(yoff)) : dem_im.shape[0], :]

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

        # Generate dem, use the georef-grid from the dem_to_align
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
            "({:.2f},{:.2f})\n".format(xoff, yoff)
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
        zoff = float(np.nanmean(final_dh))
        transform = Transformation(
            x_off=xoff,
            y_off=-yoff,  # -y_off because y_off from nk is north oriented
            z_off=zoff,
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
        grad2, grad1 = np.gradient(dem)  # in Python, x and y axis reversed

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

    def _fill_output_dict_with_coregistration_method(self):
        """
        Save the the coregistration method results on a Dict

        :return None
        """

        # Obtain unit of the bias and compute x and y biases
        unit_bias_value = self.dem_to_align.attrs["zunit"]
        dx_bias = (
            self.transform.x_off + self.cfg["estimated_initial_shift_x"]
        ) * self.dem_to_align.attrs["xres"]
        dy_bias = (
            self.transform.y_off + self.cfg["estimated_initial_shift_y"]
        ) * abs(self.dem_to_align.attrs["yres"])

        # Save nuth et kaab coregistration results
        self.demcompare_results["coregistration_results"] = {}
        self.demcompare_results["coregistration_results"]["dx"] = {
            "nuth_offset": round(self.transform.x_off, 5),
            "unit_nuth_offset": "px",
            "bias_value": round(dx_bias, 5),
            "unit_bias_value": unit_bias_value.name,
        }
        self.demcompare_results["coregistration_results"]["dy"] = {
            "nuth_offset": round(self.transform.y_off, 5),
            "unit_nuth_offset": "px",
            "bias_value": round(dy_bias, 5),
            "unit_bias_value": unit_bias_value.name,
        }

        # -> for the coordinate bounds to apply the offsets
        #    to the original DSM with GDAL
        ulx, uly, lrx, lry = compute_gdal_translate_bounds(
            self.transform.y_off,
            self.transform.x_off,
            self.dem_to_align["image"].shape,
            self.dem_to_align["georef_transform"].data,
        )
        self.demcompare_results["coregistration_results"][
            "gdal_translate_bounds"
        ] = {
            "ulx": round(ulx, 5),
            "uly": round(uly, 5),
            "lrx": round(lrx, 5),
            "lry": round(lry, 5),
        }

        # Logging report
        logging.info("# Coregistration results:")
        logging.info("\nPlanimetry 2D shift between DEM and REF:")
        logging.info(
            " -> row : {}".format(
                self.demcompare_results["coregistration_results"]["dy"][
                    "bias_value"
                ]
                * unit_bias_value
            )
        )
        logging.info(
            " -> col : {}".format(
                self.demcompare_results["coregistration_results"]["dx"][
                    "bias_value"
                ]
                * unit_bias_value
            )
        )
        logging.info("\nAltimetry shift between COREG_DEM and COREG_REF")
        logging.info(
            (" -> alti : {}".format(self.transform.z_off * unit_bias_value))
        )
