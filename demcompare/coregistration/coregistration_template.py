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
This module contains the coregistration class template.
It contains the structure for all coregistration methods in subclasses and
generic coregistration code to avoid duplication.
"""

# Standard imports
import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

# Third party imports
import numpy as np
import xarray as xr
from json_checker import And, Checker, Or

# Demcompare imports
from ..dem_tools import (
    SamplingSourceParameter,
    copy_dem,
    reproject_dems,
    save_dem,
)
from ..img_tools import compute_gdal_translate_bounds
from ..internal_typing import ConfigType
from ..transformation import Transformation


# pylint:disable=too-many-instance-attributes
class CoregistrationTemplate(metaclass=ABCMeta):
    """
    Class for general specification of a coregistration class
    """

    # Sampling source
    _SAMPLING_SOURCE: str = SamplingSourceParameter.SEC.value
    # Coreg method outputs
    _SAVE_OPTIONAL_OUTPUTS = False

    @abstractmethod
    def __init__(self, cfg: ConfigType):
        """
        Return the coregistration object associated with the method_name
        given in the configuration

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
           (such as nuth et kaab iteration plots) are saved and the internal
           dems of the coregistration
           such as reproj_dem, reproj_ref, reproj_coreg_sec,
           reproj_coreg_ref are saved.
        }

        :param cfg: configuration {'method_name': value}
        :type cfg: ConfigType
        """

        # Original sec
        self.orig_sec: xr.Dataset = None
        # Reprojected and cropped dem to align
        self.reproj_sec: xr.Dataset = None
        # Reprojected and cropped ref
        self.reproj_ref: xr.Dataset = None
        # Intermediate coregistered sec
        self.reproj_coreg_sec: xr.Dataset = None
        # Intermediate coregistered ref
        self.reproj_coreg_ref: xr.Dataset = None
        # Coregistered sec
        self.coreg_sec: xr.Dataset = None

        # Computed Transformation: result of coregistration process
        self.transform: Transformation = None
        # Offset adapting factor
        self.adapting_factor: Tuple[float, float] = (1.0, 1.0)
        # Demcompare results dict
        self.coregistration_results: Dict = None
        # Conf schema
        self.schema: Dict = None

        # Fill configuration file
        self.cfg = self.fill_conf_and_schema(cfg)
        # Check and update configuration file
        self.cfg = self.check_conf(self.cfg)

        # Initialize coregistration attributes

        # Sampling source considered during reprojection
        # (see dem_tools.SamplingSourceParameter)
        self.sampling_source = self.cfg["sampling_source"]
        # Estimated initial shift x
        self.estimated_initial_shift_x = self.cfg["estimated_initial_shift_x"]
        # Estimated initial shift y
        self.estimated_initial_shift_y = self.cfg["estimated_initial_shift_y"]
        # Save coreg_method outputs
        self.save_optional_outputs = self.cfg["save_optional_outputs"]
        # Output directory to save results
        self.output_dir = self.cfg["output_dir"]

        if self.output_dir is not None:
            # create coreg module output directory if given in configuration
            # if used in standalone, be sure that the path is absolute
            os.makedirs(cfg["output_dir"], exist_ok=True)

    @abstractmethod
    def fill_conf_and_schema(self, cfg: ConfigType = None) -> ConfigType:
        """
        Add default values to the dictionary if there are missing
        elements and define the configuration schema

        :param cfg: coregistration configuration
        :type cfg: ConfigType
        :return: cfg coregistration configuration updated
        :rtype: ConfigType
        """
        # If no cfg was given, initialize it
        if bool(cfg) is False:
            cfg = {}

        # Give the default value if the required element
        # is not in the configuration
        if "method_name" not in cfg:
            # Necessary disable to allow the default method
            cfg["method_name"] = self.method_name  # pylint:disable=no-member
        if "sampling_source" not in cfg:
            cfg["sampling_source"] = self._SAMPLING_SOURCE
        if "estimated_initial_shift_x" not in cfg:
            cfg["estimated_initial_shift_x"] = 0
            cfg["estimated_initial_shift_y"] = 0
        if "save_optional_outputs" not in cfg:
            cfg["save_optional_outputs"] = self._SAVE_OPTIONAL_OUTPUTS

        if "output_dir" not in cfg:
            cfg["output_dir"] = None
            if cfg["save_optional_outputs"]:
                raise ValueError(
                    "save_optional_outputs"
                    " option IS activated but no output_dir has been set. "
                    "Please set the output_dir parameter or deactivate"
                    " the saving options."
                )

        # Configuration schema
        self.schema = {
            "sampling_source": And(
                str,
                Or(
                    lambda input: SamplingSourceParameter.REF.value,
                    lambda input: SamplingSourceParameter.SEC.value,
                ),
            ),
            "estimated_initial_shift_x": Or(int, float),
            "estimated_initial_shift_y": Or(int, float),
            "method_name": And(str, lambda input: "nuth_kaab_internal"),
            "output_dir": Or(str, None),
            "save_optional_outputs": bool,
        }
        return cfg

    def check_conf(self, cfg: ConfigType = None) -> ConfigType:
        """
        Check if the config is correct according
        to the class configuration schema

        raises CheckerError if configuration invalid.

        :param cfg: coregistration configuration
        :type cfg: ConfigType
        :return: cfg coregistration configuration updated
        :rtype: ConfigType
        """

        checker = Checker(self.schema)
        cfg = checker.validate(cfg)
        return cfg

    def prepare_dems_for_coregistration(
        self,
        sec: xr.Dataset,
        ref: xr.Dataset,
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, Tuple[float, float]]:
        """
        Reproject the two input DEMs to the same resolution
        and size. orig_sec, reproj_sec,
        reproj_ref and the offset adapting_factor are stored as
        attributes of the class.

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
        :return: reproj_sec xr.Dataset, reproj_ref xr.Dataset,
                orig_sec xr.Dataset, adapting_factor
                Tuple[float, float].
                The xr.Datasets containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :rtype: Transformation
        """

        logging.debug("Input Coregistration SEC: %s", sec.attrs["input_img"])
        logging.debug("Input Coregistration REF: %s", ref.attrs["input_img"])
        # Store the original sec prior to reprojection
        self.orig_sec = copy_dem(sec)

        # Reproject and crop DEMs
        (
            self.reproj_sec,
            self.reproj_ref,
            self.adapting_factor,
        ) = reproject_dems(
            sec,
            ref,
            self.estimated_initial_shift_x,
            self.estimated_initial_shift_y,
            self.sampling_source,
        )
        return (
            self.reproj_sec,
            self.reproj_ref,
            self.orig_sec,
            self.adapting_factor,
        )

    def compute_coregistration(
        self,
        sec: xr.Dataset,
        ref: xr.Dataset,
    ) -> Transformation:
        """
        Reproject and compute coregistration between the two input DEMs.
        A Transformation object is returned.

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
        :return: transformation
        :rtype: Transformation
        """
        # Prepare dems for coregistration reprojecting them
        # to the same resolution and size
        (
            self.reproj_sec,
            self.reproj_ref,
            self.orig_sec,
            self.adapting_factor,
        ) = self.prepare_dems_for_coregistration(sec, ref)

        # Do coregistration
        (
            self.transform,
            self.reproj_coreg_sec,
            self.reproj_coreg_ref,
        ) = self._coregister_dems_algorithm(self.reproj_sec, self.reproj_ref)

        # Apply coregistration offsets to the original DEM and store it
        # reprojection is also done.
        self.coreg_sec = self.transform.apply_transform(sec)

        # Compute and store the coregistration_results dict
        self.save_results_dict()
        # Save internal_dems if the option was chosen
        if self.save_optional_outputs:
            self.save_internal_outputs()

        # Return the transform
        return self.transform

    @abstractmethod
    def _coregister_dems_algorithm(
        self,
        sec: xr.Dataset,
        ref: xr.Dataset,
    ) -> Tuple[Transformation, xr.Dataset, xr.Dataset]:
        """
        Coregister_dems, computes coregistration
        transform and coregistered DEMS of two DEMs
        that have the same size and resolution.

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

    def save_internal_outputs(self):
        """
        Save the dems obtained from the coregistration to .tif
        and updates its path on the coregistration_results file

        - ./coregistration/reproj_SEC.tif -> reprojected sec
        - ./coregistration/reproj_REF.tif -> reprojected ref
        - ./coregistration/reproj_coreg_SEC.tif -> reprojected
          coregistered sec
        - ./coregistration/reproj_coreg_REF.tif -> reprojected
          coregistered ref
        - ./coregistration/coreg_sec.tif -> coregistered ref

        :return: None
        """
        # Saves reprojected DEM to file system
        self.reproj_sec = save_dem(
            self.reproj_sec,
            os.path.join(self.output_dir, "reproj_SEC.tif"),
        )
        # Saves reprojected REF to file system
        self.reproj_ref = save_dem(
            self.reproj_ref,
            os.path.join(self.output_dir, "reproj_REF.tif"),
        )
        # Saves reprojected coregistered DEM to file system
        self.reproj_coreg_sec = save_dem(
            self.reproj_coreg_sec,
            os.path.join(self.output_dir, "reproj_coreg_SEC.tif"),
        )
        # Saves reprojected coregistered REF to file system
        self.reproj_coreg_ref = save_dem(
            self.reproj_coreg_ref,
            os.path.join(self.output_dir, "reproj_coreg_REF.tif"),
        )
        # Save the coregistered DEM
        self.coreg_sec = save_dem(
            self.coreg_sec,
            os.path.join(self.output_dir, "coreg_SEC.tif"),
        )
        # Update path on coregistration_results file
        if self.coregistration_results:
            self.coregistration_results["coregistration_results"][
                "reproj_coreg_ref"
            ]["path"] = self.reproj_coreg_ref.attrs["input_img"]
            # Update path on coregistration_results file
            self.coregistration_results["coregistration_results"][
                "reproj_coreg_sec"
            ]["path"] = self.reproj_coreg_sec.attrs["input_img"]

    @abstractmethod
    def save_results_dict(self):
        """
        Save the coregistration results on a Dict
        The altimetric and coregistration results are saved.
        Logging of the altimetric results is done in this function.

        :return: None
        """

        # Initialize coregistration_results dict
        self.coregistration_results = {}

        # Add coregistration_results with information regarding the
        # reprojected coregistered DEMs
        self.coregistration_results["coregistration_results"] = {}

        # Reprojected coregistered ref information
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_ref"
        ] = {}
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_ref"
        ]["path"] = self.reproj_coreg_ref.attrs["input_img"]
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_ref"
        ]["nodata"] = self.reproj_coreg_ref.attrs["nodata"]
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_ref"
        ]["nb_points"] = self.reproj_coreg_ref["image"].data.size
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_ref"
        ]["nb_valid_points"] = np.count_nonzero(
            ~np.isnan(self.reproj_coreg_ref["image"].data)
        )
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_ref"
        ]["percentage_valid_points"] = (
            self.coregistration_results["coregistration_results"][
                "reproj_coreg_ref"
            ]["nb_valid_points"]
            / self.coregistration_results["coregistration_results"][
                "reproj_coreg_ref"
            ]["nb_points"]
        ) * 100

        # Reprojected coregistered sec information
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_sec"
        ] = {}
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_sec"
        ]["path"] = self.reproj_coreg_sec.attrs["input_img"]
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_sec"
        ]["nodata"] = self.reproj_coreg_sec.attrs["nodata"]
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_sec"
        ]["nb_points"] = self.reproj_coreg_sec["image"].data.size
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_sec"
        ]["nb_valid_points"] = np.count_nonzero(
            ~np.isnan(self.reproj_coreg_sec["image"].data)
        )
        self.coregistration_results["coregistration_results"][
            "reproj_coreg_sec"
        ]["percentage_valid_points"] = (
            self.coregistration_results["coregistration_results"][
                "reproj_coreg_sec"
            ]["nb_valid_points"]
            / self.coregistration_results["coregistration_results"][
                "reproj_coreg_sec"
            ]["nb_points"]
        ) * 100

        # Obtain unit of the bias and compute x and y biases
        # use abs() to not consider the sign of x,y resolution in orig_sec
        unit_bias_value = self.orig_sec.attrs["zunit"]
        dx_bias = self.transform.total_offset_x * abs(
            self.orig_sec.attrs["xres"]
        )
        dy_bias = self.transform.total_offset_y * abs(
            self.orig_sec.attrs["yres"]
        )

        # Save coregistration offset bias results
        self.coregistration_results["coregistration_results"]["dx"] = {
            "total_offset": round(self.transform.total_offset_x, 5),
            "unit_offset": "px",
            "total_bias_value": round(dx_bias, 5),
            "unit_bias_value": unit_bias_value.name,
        }
        self.coregistration_results["coregistration_results"]["dy"] = {
            "total_offset": round(self.transform.total_offset_y, 5),
            "unit_offset": "px",
            "total_bias_value": round(dy_bias, 5),
            "unit_bias_value": unit_bias_value.name,
        }
        # for dz, directly in altitude unit, offset is directly bias (no pixel)
        self.coregistration_results["coregistration_results"]["dz"] = {
            "total_bias_value": round(self.transform.z_offset, 5),
            "unit_bias_value": unit_bias_value.name,
        }

        # -> for the coordinate bounds to apply the offsets
        #    to the original DSM with GDAL
        ulx, uly, lrx, lry = compute_gdal_translate_bounds(
            self.transform.y_offset,
            self.transform.x_offset,
            (self.orig_sec["image"].shape[0], self.orig_sec["image"].shape[1]),
            self.orig_sec["georef_transform"].data,
        )
        self.coregistration_results["coregistration_results"][
            "gdal_translate_bounds"
        ] = {
            "ulx": round(ulx, 5),
            "uly": round(uly, 5),
            "lrx": round(lrx, 5),
            "lry": round(lry, 5),
        }

        # Logging report
        logging.info("Coregistration results:")
        logging.info("Planimetry 2D shift found between reprojected REF-SEC:")
        logging.info(
            " -> row (y) : %s (%s pixels)",
            self.coregistration_results["coregistration_results"]["dy"][
                "total_bias_value"
            ]
            * unit_bias_value,
            self.coregistration_results["coregistration_results"]["dy"][
                "total_offset"
            ],
        )
        logging.info(
            " -> col (x) : %s (%s pixels)",
            self.coregistration_results["coregistration_results"]["dx"][
                "total_bias_value"
            ]
            * unit_bias_value,
            self.coregistration_results["coregistration_results"]["dx"][
                "total_offset"
            ],
        )
        logging.info("GDAL translate bounds:")
        logging.info(
            " -> ulx : %.2f , -> uly : %.2f , -> lrx : %.2f , -> lry : %.2f ",
            ulx,
            uly,
            lrx,
            lry,
        )
        logging.info(
            "Mean altimetry shift found "
            "between reprojected REF-SEC (not applied):"
        )
        logging.info(
            " -> alti : %s",
            self.coregistration_results["coregistration_results"]["dz"][
                "total_bias_value"
            ]
            * unit_bias_value,
        )
