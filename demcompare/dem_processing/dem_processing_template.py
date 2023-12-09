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
Mainly contains the DemProcessingTemplate class.
"""
# Standard imports
from abc import ABCMeta, abstractmethod
from typing import Dict

# Third party imports
import xarray as xr


class DemProcessingTemplate(
    metaclass=ABCMeta
):  # pylint:disable=too-few-public-methods
    """
    DEM processing class
    """

    def __init__(
        self, parameters: Dict = None
    ):  # pylint:disable = unused-argument
        """
        Initialization of a DEM processing object

        :param parameters: optional input parameters
        :type parameters: str
        :return: None
        """

        # DEM processing type (set to dataset)
        self.type: str = None
        # Dem processing information for report (set to dataset)
        self.fig_title: str = None
        self.colorbar_title: str = None
        self.cmap: str = None

    @abstractmethod
    def process_dem(
        self,
        dem_1: xr.Dataset,
        dem_2: xr.Dataset,
    ) -> xr.Dataset:
        """
        DEM processing method

        :param dem_1: dem_1 xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem_1: xr.Dataset
        :param dem_2: optional argument.
                      should not be given as input,
                      when the DEM processing method takes only 1 DEM as input.
                      xr.DataSet containing:

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
