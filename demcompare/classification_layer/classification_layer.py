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
Mainly contains the ClassificationLayer class.
A classification_layer defines a way to classify the DEMs alti differences.
"""

import logging
from typing import Any, Dict

import xarray as xr


class ClassificationLayer:
    """
    ClassificationLayer factory:
    A class designed for registered all available classification methods
    and instantiate them when needed.
    """

    available_classification: Dict[str, Any] = {}

    def __new__(
        cls,
        name: str,
        classification_layer_kind: str,
        cfg: Dict = None,
        dem: xr.Dataset = None,
    ):
        """
        Return a ClassificationLayerTemplate child instance
        associated with the classification_layer_kind given
        through create_classification local method for clarity.

        :param name: classification layer name
        :type name: str
        :param classification_layer_kind: classification layer kind
        :type classification_layer_kind: str
        :param cfg: layer's configuration
        :type cfg: Dict[str, Any]
        :param dem: ref dem
        :type dem:   xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        """
        return cls.create_classification(
            name,
            classification_layer_kind,
            cfg,
            dem,
        )

    @classmethod
    def create_classification(
        cls,
        name,
        classification_layer_kind,
        cfg,
        dem,
    ):
        """
        Factory command to create the classification from
        classification_layer_kind
        Return a ClassificationLayerTemplate child instance
        associated with the classification_layer_kind given

        :param name: classification layer name
        :type name: str
        :param classification_layer_kind: classification layer kind
        :type classification_layer_kind: str
        :param cfg: layer's configuration
        :type cfg: Dict[str, Any]
        :param dem: sec dem
        :type dem:   xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        """

        try:
            classification_class = cls.available_classification[
                classification_layer_kind
            ]
            classif = classification_class(
                name,
                classification_layer_kind,
                cfg,
                dem,
            )
            logging.debug(
                "ClassificationLayer of type: %s and name: %s",
                classification_layer_kind,
                name,
            )

        except KeyError:
            logging.error(
                "No classification layer type %s supported",
                classification_layer_kind,
            )
            raise

        return classif

    @classmethod
    def print_avalaible_classification_layer_type(cls):
        """
        Print all registered classification layer type
        """
        for classification_layer_type in cls.available_classification:
            print(classification_layer_type)

    @classmethod
    def register(cls, classification_layer_type: str):
        """
        Allows to register the ClassificationLayerTemplate
        subclass in the factory
        with its classification layer type through decorator

        :param classification_layer_type: the subclass type to be registered
        :type classification_layer_type: string
        """

        def decorator(classif_subclass):
            """
            Register the classification layer subclass in the
            available types

            :param classif_subclass: the subclass to be registered
            :type classif_subclass: object
            """
            cls.available_classification[
                classification_layer_type
            ] = classif_subclass
            return classif_subclass

        return decorator
