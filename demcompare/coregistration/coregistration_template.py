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
import sys
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Union

# Third party imports
import numpy as np
import xarray as xr
from json_checker import And, Checker, Or

# Demcompare imports
from ..dem_tools import (
    SamplingSourceParameter,
    compute_dems_diff,
    copy_dem,
    reproject_dems,
    save_dem,
)
from ..img_tools import compute_gdal_translate_bounds
from ..initialization import ConfigType
from ..output_tree_design import get_out_file_path
from ..transformation import Transformation


class CoregistrationTemplate(
    metaclass=ABCMeta
):  # pylint:disable=too-many-instance-attributes
    """
    Class for general specification of a coregistration class
    """

    # Coregistration configuration
    cfg: ConfigType = None

    # Name of the coregistration method
    method_name: str = None
    # Number of iterations
    iterations: int = None
    # Output directory to save results
    output_dir: str = None

    # Sampling source for the reprojection, "ref" or
    # "dem_to_align" (see dem_tools.SamplingSourceParameter)
    sampling_source: str = SamplingSourceParameter.DEM_TO_ALIGN.value

    # Estimated pixellic initial shift x
    estimated_initial_shift_x: Union[float, int] = 0.0
    # Estimated pixellic initial shift x
    estimated_initial_shift_y: Union[float, int] = 0.0
    # Original dem_to_align
    orig_dem_to_align: xr.Dataset = None
    # Reprojected and cropped dem to align
    reproj_dem_to_align: xr.Dataset = None
    # Reprojected and cropped ref
    reproj_ref: xr.Dataset = None
    # Intermediate coregistered dem_to_align
    reproj_coreg_dem_to_align: xr.Dataset = None
    # Intermediate coregistered ref
    reproj_coreg_ref: xr.Dataset = None
    # Coregistered dem_to_align
    coreg_dem_to_align: xr.Dataset = None
    # Final altitude map
    final_dh: xr.Dataset = None

    # Computed Transformation: result of coregistration process
    transform: Transformation = None
    # Offset adapting factor
    adapting_factor: Tuple[float, float] = (1.0, 1.0)

    # Demcompare results dict
    demcompare_results: Dict = None

    # If internal dems are to be saved
    save_internal_dems: bool = None
    # If coregistration method outputs are to be saved
    save_coreg_method_outputs: bool = None

    # Conf schema
    schema: Dict = None

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
           reproj_coreg_ref are saved.
        }

        :param cfg: configuration {'method_name': value}
        :type cfg: ConfigType
        """
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
        # Save internal dems
        self.save_internal_dems = self.cfg["save_internal_dems"]
        # Save coreg_method outputs
        self.save_coreg_method_outputs = self.cfg["save_coreg_method_outputs"]
        # Output directory to save results
        self.output_dir = self.cfg["output_dir"]

    @abstractmethod
    def fill_conf_and_schema(self, cfg: ConfigType = None) -> ConfigType:
        """
        Add default values to the dictionary if there are missing
        elements and define the configuration schema

        :param cfg: coregistration configuration
        :type cfg: ConfigType
        :return cfg: coregistration configuration updated
        :rtype: ConfigType
        """
        # If no cfg was given, initialize it
        if bool(cfg) is False:
            cfg = {}

        # Give the default value if the required element
        # is not in the configuration
        if "method_name" not in cfg:
            cfg["method_name"] = self.method_name
        if "sampling_source" not in cfg:
            cfg["sampling_source"] = self.sampling_source
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

        # Configuration schema
        self.schema = {
            "sampling_source": And(
                str,
                Or(
                    lambda input: SamplingSourceParameter.REF.value,
                    lambda input: SamplingSourceParameter.DEM_TO_ALIGN.value,
                ),
            ),
            "estimated_initial_shift_x": Or(int, float),
            "estimated_initial_shift_y": Or(int, float),
            "method_name": And(str, lambda input: "nuth_kaab_internal"),
            "output_dir": Or(str, None),
            "save_coreg_method_outputs": bool,
            "save_internal_dems": bool,
        }
        return cfg

    def check_conf(self, cfg: ConfigType = None) -> ConfigType:
        """
        Check if the config is correct according
        to the class configuration schema and return updated configuration

        raises CheckerError if configuration invalid.

        :param cfg: coregistration configuration
        :type cfg: ConfigType
        :return cfg: coregistration configuration updated
        :rtype: ConfigType
        """

        checker = Checker(self.schema)
        cfg = checker.validate(cfg)
        return cfg

    def prepare_dems_for_coregistration(
        self,
        dem_to_align: xr.Dataset,
        ref: xr.Dataset,
    ) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, Tuple[float, float]]:
        """
        Reproject the two input DEMs to the same resolution
        and size. orig_dem_to_align, reproj_dem_to_align,
        reproj_ref and the offset adapting_factor are stored as
        attributes of the class.

        :type dem_to_align: dem to align xr.Dataset containing

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :param ref: ref xr.Dataset containing

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :return: reproj_dem_to_align xr.Dataset, reproj_ref xr.Dataset,
                orig_dem_to_align xr.Dataset, adapting_factor
                Tuple[float, float].
                The xr.Datasets containing :

                 - image : 2D (row, col) xr.DataArray float32
                 - georef_transform: 1D (trans_len) xr.DataArray
        :rtype: Transformation
        """

        logging.info(
            "Input Coregistration DEM: {}".format(
                dem_to_align.attrs["input_img"]
            )
        )
        logging.info(
            "Input Coregistration REF: {}".format(ref.attrs["input_img"])
        )
        # Store the original dem_to_align prior to reprojection
        self.orig_dem_to_align = copy_dem(dem_to_align)

        # Reproject and crop DEMs
        (
            self.reproj_dem_to_align,
            self.reproj_ref,
            self.adapting_factor,
        ) = reproject_dems(
            dem_to_align,
            ref,
            self.estimated_initial_shift_x,
            self.estimated_initial_shift_y,
            self.sampling_source,
        )
        return (
            self.reproj_dem_to_align,
            self.reproj_ref,
            self.orig_dem_to_align,
            self.adapting_factor,
        )

    def compute_coregistration(
        self,
        dem_to_align: xr.Dataset,
        ref: xr.Dataset,
    ) -> Transformation:
        """
        Reproject and compute coregistration between the two input DEMs.
        A Transformation object is returned.

        :type dem_to_align: dem to align xr.Dataset containing

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :param ref: ref xr.Dataset containing

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :return: transformation
        :rtype: Transformation
        """
        # Prepare dems for coregistration reprojecting them
        # to the same resolution and size
        (
            self.reproj_dem_to_align,
            self.reproj_ref,
            self.orig_dem_to_align,
            self.adapting_factor,
        ) = self.prepare_dems_for_coregistration(dem_to_align, ref)

        # Do coregistration
        (
            self.transform,
            self.reproj_coreg_dem_to_align,
            self.reproj_coreg_ref,
        ) = self._coregister_dems_algorithm(
            self.reproj_dem_to_align, self.reproj_ref
        )

        # Compute and store the demcompare_results dict
        self.compute_results()
        # Save internal_dems if the option was chosen
        if self.save_internal_dems:
            self.save_internal_outputs()
        # Return the transform
        return self.transform

    @abstractmethod
    def _coregister_dems_algorithm(
        self,
        dem_to_align: xr.Dataset,
        ref: xr.Dataset,
    ) -> Tuple[Transformation, xr.Dataset, xr.Dataset]:
        """
        Coregister_dems, computes coregistration
        transform and coregistered DEMS of two DEMs
        that have the same size and resolution.

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

    def save_internal_outputs(self):
        """
        Save the dems obtained from the coregistration to .tif
        and updates its path on the demcompare_results file

            - ./coregistration/reproj_DEM.tif -> reprojected dem_to_align
            - ./coregistration/reproj_REF.tif -> reprojected ref
            - ./coregistration/reproj_coreg_DEM.tif -> reprojected
               coregistered dem_to_align
            - ./coregistration/reproj_coreg_REF.tif -> reprojected
               coregistered ref

        :return: None
        """
        # Saves reprojected DEM to file system
        self.reproj_dem_to_align = save_dem(
            self.reproj_dem_to_align,
            os.path.join(self.output_dir, get_out_file_path("reproj_DEM.tif")),
        )
        # Saves reprojected REF to file system
        self.reproj_ref = save_dem(
            self.reproj_ref,
            os.path.join(self.output_dir, get_out_file_path("reproj_REF.tif")),
        )
        # Saves reprojected coregistered DEM to file system
        self.reproj_coreg_dem_to_align = save_dem(
            self.reproj_coreg_dem_to_align,
            os.path.join(
                self.output_dir, get_out_file_path("reproj_coreg_DEM.tif")
            ),
        )
        # Saves reprojected coregistered REF to file system
        self.reproj_coreg_ref = save_dem(
            self.reproj_coreg_ref,
            os.path.join(
                self.output_dir, get_out_file_path("reproj_coreg_REF.tif")
            ),
        )
        # Update path on demcompare_results file
        if self.demcompare_results:
            self.demcompare_results["alti_results"]["reproj_coreg_ref"][
                "path"
            ] = self.reproj_coreg_ref.attrs["input_img"]
            # Update path on demcompare_results file
            self.demcompare_results["alti_results"][
                "reproj_coreg_dem_to_align"
            ]["path"] = self.reproj_coreg_dem_to_align.attrs["input_img"]
            # Update path on demcompare_results file
            self.demcompare_results["alti_results"]["dz"][
                "dz_map_path"
            ] = self.final_dh.attrs["input_img"]

    @abstractmethod
    def compute_results(self):
        """
        Save the coregistration results on a Dict
        The altimetric and coregistration results are saved.
        Logging of the altimetric results is done in this function.

        :return: None
        """

        # Initialize demcompare_results dict
        self.demcompare_results = {}

        # Add alti_results with information regarding the
        # reprojected coregistered DEMs
        self.demcompare_results["alti_results"] = {}

        # Reprojected coregistered ref information
        self.demcompare_results["alti_results"]["reproj_coreg_ref"] = {}
        self.demcompare_results["alti_results"]["reproj_coreg_ref"][
            "path"
        ] = self.reproj_coreg_ref.attrs["input_img"]
        self.demcompare_results["alti_results"]["reproj_coreg_ref"][
            "nodata"
        ] = self.reproj_coreg_ref.attrs["no_data"]
        self.demcompare_results["alti_results"]["reproj_coreg_ref"][
            "nb_points"
        ] = self.reproj_coreg_ref["image"].data.size
        self.demcompare_results["alti_results"]["reproj_coreg_ref"][
            "nb_valid_points"
        ] = np.count_nonzero(~np.isnan(self.reproj_coreg_ref["image"].data))

        # Reprojected coregistered dem_to_align information
        self.demcompare_results["alti_results"][
            "reproj_coreg_dem_to_align"
        ] = {}
        self.demcompare_results["alti_results"]["reproj_coreg_dem_to_align"][
            "path"
        ] = self.reproj_coreg_dem_to_align.attrs["input_img"]
        self.demcompare_results["alti_results"]["reproj_coreg_dem_to_align"][
            "nodata"
        ] = self.reproj_coreg_dem_to_align.attrs["no_data"]
        self.demcompare_results["alti_results"]["reproj_coreg_dem_to_align"][
            "nb_points"
        ] = self.reproj_coreg_dem_to_align["image"].data.size
        self.demcompare_results["alti_results"]["reproj_coreg_dem_to_align"][
            "nb_valid_points"
        ] = np.count_nonzero(
            ~np.isnan(self.reproj_coreg_dem_to_align["image"].data)
        )

        # Altitude difference information
        # Compute final_dh to complete the alti_resuts
        self.final_dh = compute_dems_diff(
            self.reproj_coreg_ref, self.reproj_coreg_dem_to_align
        )
        self.demcompare_results["alti_results"]["dz"] = {}
        self.demcompare_results["alti_results"]["dz"] = {
            "dz_map_path": self.final_dh.attrs["input_img"],
            "total_bias_value": round(
                float(np.nanmean(self.final_dh["image"].data)), 5
            ),
            "zunit": self.reproj_coreg_dem_to_align.attrs["zunit"].name,
            "percent": round(
                100
                * np.count_nonzero(~np.isnan(self.final_dh["image"].data))
                / self.final_dh["image"].data.size,
                5,
            ),
            "nodata": self.final_dh.attrs["no_data"],
            "nb_points": self.final_dh["image"].data.size,
            "nb_valid_points": np.count_nonzero(
                ~np.isnan(self.final_dh["image"].data)
            ),
        }

        # Obtain unit of the bias and compute x and y biases
        unit_bias_value = self.orig_dem_to_align.attrs["zunit"]
        dx_bias = (
            self.transform.total_offset_x * self.orig_dem_to_align.attrs["xres"]
        )
        dy_bias = (
            self.transform.total_offset_y * self.orig_dem_to_align.attrs["xres"]
        )

        # Save coregistration results
        self.demcompare_results["coregistration_results"] = {}
        self.demcompare_results["coregistration_results"]["dx"] = {
            "total_offset": round(self.transform.total_offset_x, 5),
            "unit_offset": "px",
            "total_bias_value": round(dx_bias, 5),
            "unit_bias_value": unit_bias_value.name,
        }
        self.demcompare_results["coregistration_results"]["dy"] = {
            "total_offset": round(self.transform.total_offset_y, 5),
            "unit_offset": "px",
            "total_bias_value": round(dy_bias, 5),
            "unit_bias_value": unit_bias_value.name,
        }

        # -> for the coordinate bounds to apply the offsets
        #    to the original DSM with GDAL
        ulx, uly, lrx, lry = compute_gdal_translate_bounds(
            self.transform.y_offset,
            self.transform.x_offset,
            self.orig_dem_to_align["image"].shape,
            self.orig_dem_to_align["georef_transform"].data,
        )
        self.demcompare_results["coregistration_results"][
            "gdal_translate_bounds"
        ] = {
            "ulx": round(ulx, 5),
            "uly": round(uly, 5),
            "lrx": round(lrx, 5),
            "lry": round(lry, 5),
        }

        # -> for the coordinate bounds to apply the offsets
        #    to the original DSM with GDAL
        ulx, uly, lrx, lry = compute_gdal_translate_bounds(
            self.transform.y_offset,
            self.transform.x_offset,
            self.orig_dem_to_align["image"].shape,
            self.orig_dem_to_align["georef_transform"].data,
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
        logging.info("Planimetry 2D shift between DEM and REF:")
        logging.info(
            " -> row : {}".format(
                self.demcompare_results["coregistration_results"]["dy"][
                    "total_bias_value"
                ]
                * unit_bias_value
            )
        )
        logging.info(
            " -> col : {}".format(
                self.demcompare_results["coregistration_results"]["dx"][
                    "total_bias_value"
                ]
                * unit_bias_value
            )
        )
        logging.info("Altimetry shift between COREG_DEM and COREG_REF:")
        logging.info(
            (" -> alti : {}".format(self.transform.z_offset * unit_bias_value))
        )
