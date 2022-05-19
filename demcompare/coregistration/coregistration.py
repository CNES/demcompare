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
This module contains classes and functions associated to the DEM coregistration.
"""
import logging
import os
import sys
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np
import xarray as xr

from ..dem_tools import (
    SamplingSourceParameter,
    compute_dems_diff,
    copy_dem,
    reproject_dems,
    save_dem,
)
from ..output_tree_design import get_out_file_path
from ..transformation import Transformation


class Coregistration(metaclass=ABCMeta):
    """
    Coregistration class
    """

    coregistration_methods_avail: Dict = {}

    # Coregistration configuration
    cfg: Dict = None

    # Name of the coregistration method
    method_name: str = None

    # Output directory to save results
    output_dir: str = None

    # Sampling source for the reprojection, "ref" or
    # "dem_to_align" (see dem_tools.SamplingSourceParameter)
    sampling_source: SamplingSourceParameter = None

    # Estimated pixellic initial shift x
    estimated_initial_shift_x: Union[float, int] = None
    # Estimated pixellic initial shift x
    estimated_initial_shift_y: Union[float, int] = None

    # Original dem_to_align
    dem_to_align: xr.Dataset = None
    # Initial and final altitude difference
    initial_dh: xr.Dataset = None
    final_dh: xr.Dataset = None
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

    # Computed Transform
    transform: Transformation = None

    # Demcompare results dict
    demcompare_results: Dict = None

    # If internal dems are to be saved
    save_internal_dems: bool = None
    # If coregistration method outputs are to be saved
    save_coreg_method_outputs: bool = None

    def __new__(cls, **cfg: dict):
        """
        Return the plugin associated with the method_name
        given in the configuration

        :param cfg: configuration {'method_name': value}
        :type cfg: dictionary
        """
        if cls is Coregistration:
            if isinstance(cfg["method_name"], str):
                try:
                    return super(Coregistration, cls).__new__(
                        cls.coregistration_methods_avail[cfg["method_name"]]
                    )
                except KeyError:
                    logging.error(
                        "No coregistration method named {} supported".format(
                            cfg["method_name"]
                        )
                    )
                    sys.exit(1)
        else:
            return super(Coregistration, cls).__new__(cls)
        return None

    # Add the pylint disable to allow the optional
    # arguments (in this case method_name)
    def __init_subclass__(
        cls, method_name: str
    ):  # pylint:disable=unexpected-special-method-signature
        cls.method_name = method_name
        super().__init_subclass__()
        cls.coregistration_methods_avail[method_name] = cls

    def compute_coregistration(
        self,
        dem_to_align: xr.Dataset,
        ref: xr.Dataset,
    ) -> Transformation:
        """
        Reproject and compute coregistration and initial
        and final altitude difference between the two input DEMs.
        A Transformation object is returned.

        :type dem_to_align: dem to align xr.Dataset containing

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :param ref: ref xr.Dataset containing

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :return: transformation, initial_dh xr.Dataset, final_dh xr.Dataset,
                sampling_source, results_dict. The xr.Datasets containing :

                 - image : 2D (row, col) xr.DataArray float32
                 - georef_transform: 1D (trans_len) xr.DataArray
        :rtype: Transformation
        """

        logging.info("DEM: {}".format(dem_to_align.attrs["input_img"]))
        logging.info("REF: {}".format(ref.attrs["input_img"]))
        # Store the original dem_to_align prior to reprojection
        self.dem_to_align = copy_dem(dem_to_align)

        # Reproject and crop DEMs
        (
            self.reproj_dem_to_align,
            self.reproj_ref,
            adapting_factor,
        ) = reproject_dems(
            dem_to_align,
            ref,
            self.estimated_initial_shift_x,
            self.estimated_initial_shift_y,
            self.sampling_source,
        )

        # Compute initial_dh
        self.initial_dh = compute_dems_diff(
            self.reproj_ref, self.reproj_dem_to_align
        )

        # Do coregistration
        (
            self.transform,
            self.reproj_coreg_dem_to_align,
            self.reproj_coreg_ref,
        ) = self._coregister_dems(self.reproj_dem_to_align, self.reproj_ref)

        # Adapt the transformation to the dem_to_align sampling
        if self.sampling_source == "ref":
            self.transform.adapt_transform_offset(adapting_factor)

        # Apply coregistration offsets to the original DEM and store it
        self.coreg_dem_to_align = self.transform.apply_transform(dem_to_align)

        # Compute final_dh
        self.final_dh = compute_dems_diff(
            self.reproj_coreg_ref, self.reproj_coreg_dem_to_align
        )

        # Compute and store the demcompare_results dict
        self.compute_results_dict()

        # Return the transform
        return self.transform

    @abstractmethod
    def _coregister_dems(
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

    @abstractmethod
    def _fill_output_dict_with_coregistration_method(self):
        """
        Save the coregistration method results on a Dict

        :return: None
        """

    def save_outputs_to_disk(self):
        """
        Save the dems obtained from the coregistration to .tif

        The saved DEMs are the following:
        - coreg_DEM.tif -> coregistered dem_to_align

        if save_internal_dems is set to True :
            - initial_dh.tif -> initial altitude difference
               (interm_DEM - interm_REF)
            - final_dh.tif -> final altitude difference
               (interm_coreg_DEM - interm_coreg_REF)
            - reproj_DEM.tif -> reprojected coregistered dem_to_align
            - reproj_REF.tif -> reprojected ctransformoregistered ref
            - reproj_coreg_DEM.tif -> reprojected coregistered dem_to_align
            - reproj_coreg_REF.tif -> reprojected coregistered ref

        :return: None
        """

        # Save coregistered DEM
        # - coreg_DEM.tif -> coregistered dem_to_align

        save_dem(
            self.coreg_dem_to_align,
            os.path.join(self.output_dir, get_out_file_path("coreg_DEM.tif")),
        )

        # Save internal_dems if the option was choosen
        if self.save_internal_dems:
            # Saves reprojected DEM to file system
            self.reproj_dem_to_align = save_dem(
                self.reproj_dem_to_align,
                os.path.join(
                    self.output_dir, get_out_file_path("reproj_DEM.tif")
                ),
            )
            # Saves reprojected REF to file system
            self.reproj_ref = save_dem(
                self.reproj_ref,
                os.path.join(
                    self.output_dir, get_out_file_path("reproj_REF.tif")
                ),
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
            # Saves initial altitude difference to file system
            self.initial_dh = save_dem(
                self.initial_dh,
                os.path.join(
                    self.output_dir, get_out_file_path("initial_dh.tif")
                ),
            )
            # Saves final altitude difference to file system
            self.final_dh = save_dem(
                self.final_dh,
                os.path.join(
                    self.output_dir, get_out_file_path("final_dh.tif")
                ),
            )
            if self.demcompare_results:
                # Update path on demcompare_results file
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

    def compute_results_dict(self):
        """
        Save the coregistration results on a Dict
        The altimetric results are saved by the abstract class,
        and the coregistration_results depending on the coregistration method
        are saved by the subclass.

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
        self.demcompare_results["alti_results"]["dz"] = {}
        self.demcompare_results["alti_results"]["dz"] = {
            "dz_map_path": self.final_dh.attrs["input_img"],
            "bias_value": round(
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

        # Save coregistration method results on demcompare_results
        self._fill_output_dict_with_coregistration_method()

    def get_results(self) -> Tuple[xr.Dataset, Dict]:
        """
        Returns the main outputs of the coregistration:
        - The coregistered input dem_to_align
        - The demcompare_results dict with the coregistration results

        :return: coreg_dem_to_align xr.Dataset, demcompare_results.
                The xr.Datasets containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :rtype: Tuple[xr.Dataset, Dict]
        """
        return self.coreg_dem_to_align, self.demcompare_results

    def get_internal_results(
        self,
    ) -> Tuple[
        xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset
    ]:
        """
        Returns the internal outputs of the coregistration:
        - The initial_dh: initial altitude difference
        - The final_dh: final altitude difference
        - The reproj_dem_to_align: reprojected dem_to_align
        - The reproj_ref: reprojected reference
        - The reproj_coreg_dem_to_align: reprojected and
        coregistered dem_to_align
        - The reproj_coreg_ref: reprojected and coregistered ref

        :return: initial_dh, final_dh, reproj_dem_to_align,
                reproj_ref, reproj_coreg_dem_to_align, reproj_coreg_ref.
                All xr.Datasets containing :

                 - im : 2D (row, col) xarray.DataArray float32
                 - trans: 1D (trans_len) xarray.DataArray
        :rtype: Tuple[xr.Dataset, xr.Dataset, xr.Dataset,
                 xr.Dataset, xr.Dataset, xr.Dataset]
        """
        return (
            self.initial_dh,
            self.final_dh,
            self.reproj_dem_to_align,
            self.reproj_ref,
            self.reproj_coreg_dem_to_align,
            self.reproj_coreg_ref,
        )
