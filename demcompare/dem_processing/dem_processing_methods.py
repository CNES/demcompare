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

# Third party imports
import xarray as xr

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
