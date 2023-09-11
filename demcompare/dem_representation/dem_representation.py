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
Mainly contains the DEM representation class.
"""

import logging
from typing import Any, Dict, Union


class DemRepresentation:
    """
    DEM representation method factory:
    A class designed for registered all available DEM representation methods
    and instantiate them when needed.
    """

    available_dem_representation_methods: Dict[str, Any] = {}

    def __new__(
        cls, dem_representation_method: str, params: Union[Dict, None] = None
    ):
        """
        Return a DemRepresentationTemplate child instance
        associated with the "dem_representation_method"
        given in the configuration through
        dem_representation_method local method for clarity.

        :param dem_representation_method: DEM representation method
        :type dem_representation_method: str
        """
        return cls.create_dem_representation_method(
            dem_representation_method, params
        )

    @classmethod
    def create_dem_representation_method(
        cls, dem_representation_method: str, parameters: Dict = None
    ):
        """
        Factory command to create the DEM representation method from method_name
        Return a DemRepresentationTemplate child instance
        associated with the name given in the configuration

        :param dem_representation_method: DEM representation method
        :type dem_representation_method: str
        :param parameters: optional input parameters
        :type parameters: Dict
        """
        try:
            dem_representation_class = cls.available_dem_representation_methods[
                dem_representation_method
            ]
            dem_representation = dem_representation_class(parameters=parameters)
        except KeyError:
            logging.error(
                "No DEM representation type %s supported",
                dem_representation_method,
            )
            raise
        return dem_representation

    @classmethod
    def print_available_dem_representation_methods(cls):
        """
        Print all registered DEM representation methods
        """
        for (
            dem_representation_method
        ) in cls.available_dem_representation_methods:
            print(dem_representation_method)

    @classmethod
    def register(cls, dem_representation_method: str):
        """
        Allows to register the DemRepresentationTemplate
        subclass in the factory
        with its DEM representation type through decorator

        :param dem_representation_method: the subclass type to be registered
        :type dem_representation_method: string
        """

        def decorator(dem_representation_subclass):
            """
            Register the DEM representation subclass in the available methods

            :param dem_representation_subclass: the subclass to be registered
            :type dem_representation_subclass: object
            """
            cls.available_dem_representation_methods[
                dem_representation_method
            ] = dem_representation_subclass
            return dem_representation_subclass

        return decorator
