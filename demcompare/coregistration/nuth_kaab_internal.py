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
# pylint:disable=too-many-lines
# Standard imports
import logging
import os
from typing import Any, Dict, Tuple, Union

# Third party imports
import matplotlib.pyplot as pl
import numpy as np
import xarray as xr
from json_checker import And
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import leastsq

# Demcompare imports
from ..dem_tools import DEFAULT_NODATA, create_dem
from ..img_tools import compute_gdal_translate_bounds
from ..output_tree_design import get_out_dir
from ..transformation import Transformation
from .coregistration import Coregistration
from .coregistration_template import CoregistrationTemplate


@Coregistration.register("nuth_kaab_internal")
class NuthKaabInternal(
    CoregistrationTemplate
):  # pylint:disable=abstract-method
    """
    NuthKaab class, allows to perform a Nuth & Kaab coregistration
    from authors above and adapted in demcompare
    """

    # Default parameters in case they are not specified in the cfg
    DEFAULT_ITERATIONS = 6
    # Method name
    method_name = "nuth_kaab_internal"

    def __init__(self, cfg: Dict[str, Any] = None):
        """
        Any coregistration class should have the following schema on
        its input cfg (optional parameters may be added for a
        particular coregistration class):

        coregistration = {
         "method_name": coregistration class name. str,
         "number_of_iterations": number of iterations. int,
         "sampling_source": optional. sampling source at which
           the dems are reprojected prior to coregistration. str
           "sec" (default) or "ref",
         "estimated_initial_shift_x": optional. estimated initial
           x shift. int or float. 0 by default,
         "estimated_initial_shift_y": optional. estimated initial
           y shift. int or float. 0 by default,
         "output_dir": optional output directory. str. If given,
           the coreg_sec is saved,
         "save_optional_outputs": optional. bool. Requires output_dir
           to be set. If activated, the outputs of the coregistration method
           (such as nuth et kaab iteration plots) are saved and
           the internal dems of the coregistration
           such as reproj_dem, reproj_ref, reproj_coreg_sec,
           reproj_coreg_ref, initial_dh and final_dh are saved.
        }

        :param cfg: configuration
        :type cfg: Dict[str, Any]
        """
        # Call generic init before supercharging
        super().__init__(cfg)
        # Number of iterations specific to Nuth et kaab internal algorithm
        self.iterations = self.cfg["number_of_iterations"]
        # Aspect bounds for the Nuth et kaab internal algorithm
        self.aspect_bounds: Union[np.ndarray, None] = None

    def fill_conf_and_schema(
        self, cfg: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Add default values to the dictionary if there are missing
        elements and define the configuration schema

        :param cfg: coregistration configuration
        :type cfg: Dict[str, Any]
        :return cfg: coregistration configuration updated
        :rtype: Dict[str, Any]
        """
        # Call generic fill_conf_and_schema
        cfg = super().fill_conf_and_schema(cfg)

        # Give the default value if the required element
        # is not in the configuration
        if "method_name" not in cfg:
            cfg["method_name"] = self.method_name
        if "number_of_iterations" not in cfg:
            cfg["number_of_iterations"] = self.DEFAULT_ITERATIONS

        # Add subclass parameter to the default schema
        self.schema["number_of_iterations"] = And(
            int, lambda input: input < 16, lambda input: input > 0
        )
        return cfg

    def _coregister_dems_algorithm(  # pylint:disable=too-many-locals
        self,
        sec: xr.Dataset,
        ref: xr.Dataset,
    ) -> Tuple[Transformation, xr.Dataset, xr.Dataset]:
        """
        Coregister_dems_algorithm, computes coregistration
        transformation and reprojected coregistered DEMs
        with Nuth et kaab algorithm
        Plots might be saved if save_optional_outputs is set.

        :param sec: sec xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type sec: xarray Dataset
        :param ref: ref xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type ref: xarray Dataset
        :return: transformation, reproj_coreg_sec xr.DataSet,
                 reproj_coreg_ref xr.DataSet. The xr.Datasets containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :rtype: Tuple[Transformation, xr.Dataset, xr.Dataset]
        """
        # Copy dataset and extract image array
        sec_im = sec["image"].data
        ref_im = ref["image"].data

        # Set target dem grid for interpolation purpose
        xgrid = np.arange(sec_im.shape[1])
        ygrid = np.arange(sec_im.shape[0])
        # Set spline interpolation
        spline_1, spline_2 = self.interpolate_dem_on_grid(ref_im, xgrid, ygrid)

        # Compute inital_dh and initialize ref
        initial_dh = ref_im - sec_im
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
        if self.save_optional_outputs:
            output_dir_ = os.path.join(
                self.output_dir, get_out_dir("nuth_kaab_tmp_dir")
            )
            pl.savefig(
                os.path.join(output_dir_, "ElevationDiff_BeforeCoreg.png"),
                dpi=100,
                bbox_inches="tight",
            )
        pl.close()
        # Initialize offsets
        x_offset, y_offset = 0.0, 0.0
        logging.debug("Nuth & Kaab iterations: %s", self.iterations)
        coreg_sec = sec_im

        # Compute bounds for different aspect slices
        self.aspect_bounds = np.arange(0, 2 * np.pi, np.pi / 36)
        for i in range(self.iterations):
            # Remove bias from ref
            coreg_ref -= median
            # Compute new elevation difference
            dh = coreg_sec - coreg_ref
            # Compute slope and aspect
            slope, aspect = self._grad2d(coreg_sec)

            if self.save_optional_outputs:
                output_dir_ = os.path.join(
                    self.output_dir, get_out_dir("nuth_kaab_tmp_dir")
                )
                plotfile = os.path.join(output_dir_, f"nuth_kaab_iter#{i}.png")
            else:
                plotfile = None

            # Compute offset
            east, north, z = self._nuth_kaab_single_iter(
                dh, slope, aspect, plot_file=plotfile
            )

            logging.debug(
                "# %s - Offset in pixels : ( %.2f , %.2f ), -bias : ( %.2f )",
                i + 1,
                east,
                north,
                z,
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

            # Crop dems with offset
            coreg_ref = self.crop_dem_with_offset(znew, x_offset, y_offset)
            coreg_sec = self.crop_dem_with_offset(sec_im, x_offset, y_offset)

            # Logging of some statistics
            diff = coreg_ref - coreg_sec
            diff = diff[np.isfinite(diff)]
            nmad_new = 1.4826 * np.median(np.abs(diff - np.median(diff)))
            median = np.median(diff)

            logging.debug(
                "\t Median : %.2f, NMAD = %.2f",
                median,
                nmad_new,
            )
            # with same dems test, nmad_old divive by zero. don't show gain
            if nmad_old != 0:
                logging.debug(
                    "\t Gain : %.2f",
                    (nmad_new - nmad_old) / nmad_old * 100,
                )
            # put new nmad to old for next iteration
            nmad_old = nmad_new

        # Initialize coregistered classification layers
        coreg_ref_classif = None
        coreg_sec_classif = None
        # If classification layers in ref, interpolate and crop them
        # To have the same modifications as ref
        if "indicator" in ref.coords:
            coreg_ref_classif = self.interpolate_classif_layers(
                ref.classification_layer_masks, xgrid, ygrid, x_offset, y_offset
            )
            coreg_ref_classif = self.crop_classif_layers(
                coreg_ref_classif, x_offset, y_offset
            )
        if "indicator" in sec.coords:
            coreg_sec_classif = self.crop_classif_layers(
                sec.classification_layer_masks, x_offset, y_offset
            )

        reproj_bounds = compute_gdal_translate_bounds(
            y_offset,
            x_offset,
            (coreg_sec.shape[0], coreg_sec.shape[1]),
            sec.georef_transform.data,
        )

        # Generate the dataset dems
        coreg_sec_dataset = create_dem(
            coreg_sec,
            transform=sec.georef_transform.data,
            nodata=DEFAULT_NODATA,
            img_crs=sec.crs,
            classification_layer_masks=coreg_sec_classif,
            bounds=reproj_bounds,
        )
        coreg_ref_dataset = create_dem(
            coreg_ref,
            transform=sec.georef_transform.data,
            nodata=DEFAULT_NODATA,
            img_crs=sec.crs,
            classification_layer_masks=coreg_ref_classif,
            bounds=reproj_bounds,
        )
        logging.debug(
            "Nuth & Kaab Final Offset in pixels (east, north): ( %.2f , %.2f )",
            x_offset,
            y_offset,
        )
        # Display
        final_dh = coreg_ref - coreg_sec
        median = np.median(final_dh[np.isfinite(final_dh)])
        nmad_old = 1.4826 * np.median(
            np.abs(final_dh[np.isfinite(final_dh)] - median)
        )
        maxval = 3 * nmad_old
        pl.figure(1, figsize=(7.0, 8.0))
        pl.imshow(final_dh, vmin=-maxval, vmax=maxval)
        color_bar = pl.colorbar()
        color_bar.set_label("Elevation difference (m)")
        if self.save_optional_outputs:
            output_dir_ = os.path.join(
                self.output_dir, get_out_dir("nuth_kaab_tmp_dir")
            )
            pl.savefig(
                os.path.join(output_dir_, "ElevationDiff_AfterCoreg.png"),
                dpi=100,
                bbox_inches="tight",
            )
        pl.close()
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
        return transform, coreg_sec_dataset, coreg_ref_dataset

    @staticmethod
    def interpolate_dem_on_grid(
        interp_dem: np.ndarray, xgrid: np.ndarray, ygrid: np.ndarray
    ) -> Tuple[RectBivariateSpline, RectBivariateSpline]:
        """
        interpolate_dem_on_grid, returns the RectBivariateSpline function
        to do Bivariate spline approximation over a rectangular mesh.

        :param interp_dem: input dem image
        :type interp_dem: np.ndarray
        :param xgrid: x axis grid
        :type xgrid: np.ndarray
        :param ygrid: x axis grid
        :type ygrid: np.ndarray
        :return: spline_1, spline_2,
        :rtype: Tuple[RectBivariateSpline, RectBivariateSpline]
        """
        # Mask nan values to -9999
        nan_maskval = np.isnan(interp_dem)
        sec_from_filled = np.where(nan_maskval, -9999, interp_dem)
        # Compute both splines
        spline_1 = RectBivariateSpline(
            ygrid, xgrid, sec_from_filled, kx=1, ky=1
        )
        spline_2 = RectBivariateSpline(ygrid, xgrid, nan_maskval, kx=1, ky=1)

        return spline_1, spline_2

    def interpolate_classif_layers(
        self,
        dem_classif: xr.DataArray,
        xgrid: np.ndarray,
        ygrid: np.ndarray,
        x_offset: float,
        y_offset: float,
    ):
        """
        interpolates the classification layers on the input
        grids with the input offsets.

        :param dem_classif: input dem image
        :type dem_classif: xr.Dataarray
        :param xgrid: x axis grid
        :type xgrid: np.ndarray
        :param ygrid: x axis grid
        :type ygrid: np.ndarray
        :param x_offset: x offset
        :type x_offset: float
        :param y_offset: y offset
        :type y_offset: float
        :return: interpolated classification layers
        :rtype: xr.Dataarray
        """
        interp_classif = dem_classif.data
        # For each existing classification layer
        for idx in range(len(dem_classif.coords["indicator"])):
            # Get classification data
            classif_data = dem_classif.data[:, :, idx]
            # Compute classif layer splines
            spline_1, spline_2 = self.interpolate_dem_on_grid(
                classif_data, xgrid, ygrid
            )
            # Apply spline interpolations
            rectified_map = spline_1(ygrid - y_offset, xgrid + x_offset)
            nanval_new = spline_2(ygrid - y_offset, xgrid + x_offset)
            rectified_map[nanval_new != 0] = np.nan
            interp_classif[:, :, idx] = rectified_map
        # Update dataset's classification data
        dem_classif.data = interp_classif
        return dem_classif

    @staticmethod
    def crop_dem_with_offset(
        dem: np.ndarray, x_offset: float, y_offset: float
    ) -> np.ndarray:
        """
        Crops the input dem with the given offsets.

        :param dem: input dem image
        :type dem: np.ndarray
        :param x_offset: x offset
        :type x_offset: float
        :param y_offset: y offset
        :type y_offset: float
        :return: cropped dem
        :rtype: np.ndarray
        """
        # Crop DEMs with offset
        if x_offset >= 0:
            cropped_dem = dem[:, 0 : dem.shape[1] - int(np.ceil(x_offset))]

        else:
            cropped_dem = dem[:, int(np.floor(-x_offset)) : dem.shape[1]]

        if -y_offset >= 0:
            cropped_dem = cropped_dem[
                0 : dem.shape[0] - int(np.ceil(-y_offset)), :
            ]

        else:
            cropped_dem = cropped_dem[int(np.floor(y_offset)) : dem.shape[0], :]

        return cropped_dem

    def crop_classif_layers(
        self, dem_classif: xr.DataArray, x_offset, y_offset
    ) -> xr.DataArray:
        """
        crop_classif_layers crops and updates the input classification layers
        with the input offsets.

        :param dem_classif: classification layers
        :type dem_classif: xr.Dataarray
        :param x_offset: x offset
        :type x_offset: float
        :param y_offset: y offset
        :type y_offset: float
        :return: cropped classification layers
        :rtype: xr.Dataarray
        """
        # Initialize cropped data
        cropped_classif_list = []
        # For each existing classification layer
        for idx in range(len(dem_classif.coords["indicator"])):
            # Get classification data
            classif_map = dem_classif.data[:, :, idx]
            # Crop classification data
            rectified_map = self.crop_dem_with_offset(
                classif_map, x_offset, y_offset
            )
            cropped_classif_list.append(rectified_map)
        # Set cropped data to the correct dimension order
        cropped_classif = np.swapaxes(
            np.transpose(np.array(cropped_classif_list)), 0, 1
        )
        # Get all classif indicators
        indicator = list(dem_classif.coords["indicator"].data)
        # Initialize new cropped classif coordinates
        coords_classification_layers = {
            "row": np.arange(rectified_map.shape[0]),
            "col": np.arange(rectified_map.shape[1]),
            "indicator": indicator,
        }
        # Create new xarray with the cropped classif
        cropped_classifs = xr.DataArray(
            data=cropped_classif,
            coords=coords_classification_layers,
            dims=["row", "col", "indicator"],
        )

        return cropped_classifs

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

        :param dh: elevation difference sec - ref
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
                str(plot_file),
                aspect,
                target,
                target_filt,
                slice_filt_median,
                yfit,
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

        :param aspect: elevation difference sec - ref
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

    def save_results_dict(self):
        """
        Save the coregistration results on a Dict
        The altimetric and coregistration results are saved.
        Logging of the altimetric results is done in this function.

        :return: None
        """
        # Call generic save_results_dict before supercharging
        super().save_results_dict()
        # Add Nuth offsets to demcompare_results
        self.demcompare_results["coregistration_results"]["dx"][
            "nuth_offset"
        ] = round(self.transform.x_offset, 5)
        self.demcompare_results["coregistration_results"]["dy"][
            "nuth_offset"
        ] = round(self.transform.y_offset, 5)
