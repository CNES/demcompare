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
This module contains classes and functions associated to the dem transformation.
"""

from typing import List, Tuple

import xarray as xr

from .dem_tools import translate_dem


class Transformation:
    """
    Transformation class
    A transformation defines a way to transform
    the DEMs by offsets and/or rotations.
    """

    #
    # Initialization
    #
    def __init__(
        self,
        x_off: float,
        y_off: float,
        z_off: float,
        rotation: List[float] = None,
    ):
        """
        Initialization of a transformation object
        """

        self.x_off = x_off
        self.y_off = y_off
        self.z_off = z_off
        self.rotation = rotation

    def apply_transform(self, dem: xr.Dataset) -> xr.Dataset:
        """
        Apply Transformation to input dem, currently only
        the offsets are considered
        # TODO: apply rotation when exists.

        :param dem: dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :type dem: xr.Dataset
        :return: transformed dem xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :rtype: xr.Dataset
        """
        # Negative because computed y_off is positive towards the north
        transformed_dem = translate_dem(dem, self.x_off, self.y_off)
        return transformed_dem

    def adapt_transform_offset(
        self, adapting_factor: Tuple[float, float]
    ) -> None:
        """
        Adapt the transform offset according to the input adapting factor

        :param adapting_factor: x and y adapting factors
        :type adapting_factor: Tuple[float, float]
        :return: None
        """
        x_factor, y_factor = adapting_factor
        self.x_off = self.x_off * x_factor
        self.y_off = self.y_off * y_factor
