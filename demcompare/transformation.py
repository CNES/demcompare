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
This module contains classes and functions associated to the dem transformation.
"""

# Standard imports
from typing import List, Tuple

# Third Party imports
import xarray as xr

# Demcompare imports
from .dem_tools import translate_dem


class Transformation:
    """
    Transformation class
    A transformation defines a way to transform
    the DEMs by offsets and/or rotations.
    For now, only x,y offset translation
    """

    def __init__(
        self,
        x_offset: float,
        y_offset: float,
        z_offset: float,
        estimated_initial_shift_x: float = 0.0,
        estimated_initial_shift_y: float = 0.0,
        adapting_factor: Tuple[float, float] = (1.0, 1.0),
        rotation: List[float] = None,
    ):
        """
        Initialization of a transformation object

        :param x_offset: pixellic x offset
        :type x_offset: float
        :param y_offset: pixellic y offset
        :type y_offset: float
        :param z_offset: pixellic z offset
        :type z_offset: float
        :param estimated_initial_shift_x: estimated initial shift x
        :type estimated_initial_shift_x: float
        :param estimated_initial_shift_y: estimated initial shift y
        :type estimated_initial_shift_y: float
        :param adapting_factor: adapting factor to adapt the
          offsets to the correct resolution
        :type adapting_factor: Tuple[float, float]
        :param rotation: rotation parameters (to be defined)
        :type rotation: List[float]
        """

        # adapt the offsets to the correct resolution with
        # the input adapting factor
        # (necessary in case the sampling_value is dem_1, otherwise
        # the adapting factor is (1.0, 1.0))
        self.adapting_factor = adapting_factor
        x_factor, y_factor = adapting_factor
        # x pixellic offset
        self.x_offset = x_offset * x_factor
        # y pixellic offset
        self.y_offset = y_offset * y_factor
        # z pixellic offset
        self.z_offset = z_offset

        # Compute the total offsets considering the estimated initial shifts
        # total offset x
        self.total_offset_x = x_offset * x_factor + estimated_initial_shift_x
        # total offset y
        self.total_offset_y = y_offset * y_factor + estimated_initial_shift_y
        # rotation
        self.rotation = rotation

    def __repr__(self):
        """
        Represent transformation offsets
        """

        output_string = (
            f"Transformation(x_offset = {round(self.x_offset, 5)},"
            + f" y_offset = {round(self.y_offset, 5)},"
            + f" z_offset = {round(self.z_offset, 5)})"
        )
        return output_string

    def apply_transform(self, dem: xr.Dataset) -> xr.Dataset:
        """
        Apply Transformation to input dem, currently only
        the offsets are considered

        :param dem: dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :type dem: xr.Dataset
        :return: transformed dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :rtype: xr.Dataset
        """
        # for this version of transform, (x,y) planimetric translation only
        transformed_dem = translate_dem(dem, self.x_offset, self.y_offset)
        return transformed_dem
