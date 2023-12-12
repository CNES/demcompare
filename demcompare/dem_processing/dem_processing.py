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
Mainly contains the DEM processing class.
"""

import logging
from typing import Any, Dict, Union


class DemProcessing:
    """
    DEM processing method factory:
    A class designed for registered all available DEM processing methods
    and instantiate them when needed.
    """

    available_dem_processing_methods: Dict[str, Any] = {}

    def __new__(
        cls, dem_processing_method: str, params: Union[Dict, None] = None
    ):
        """
        Return a DemProcessingTemplate child instance
        associated with the "dem_processing_method" given in the configuration
        through dem_processing_method local method for clarity.

        :param dem_processing_method: DEM processing method
        :type dem_processing_method: str
        """
        return cls.create_dem_processing_method(dem_processing_method, params)

    @classmethod
    def create_dem_processing_method(
        cls, dem_processing_method: str, parameters: Dict = None
    ):
        """
        Factory command to create the DEM processing method from method_name
        Return a DemProcessingTemplate child instance
        associated with the name given in the configuration

        :param dem_processing_method: DEM processing method
        :type dem_processing_method: str
        :param parameters: optional input parameters
        :type parameters: Dict
        """
        try:
            dem_processing_class = cls.available_dem_processing_methods[
                dem_processing_method
            ]
            dem_processing = dem_processing_class(parameters=parameters)
        except KeyError:
            logging.error(
                "No DEM processing type %s supported", dem_processing_method
            )
            raise
        return dem_processing

    @classmethod
    def print_dem_processing_methods(cls):
        """
        Print all registered DEM processing methods
        """
        for dem_processing_method in cls.available_dem_processing_methods:
            print(dem_processing_method)

    @classmethod
    def register(cls, dem_processing_method: str):
        """
        Allows to register the DemProcessingTemplate
        subclass in the factory
        with its DEM processing type through decorator

        :param dem_processing_method: the subclass type to be registered
        :type dem_processing_method: string
        """

        def decorator(dem_processing_subclass):
            """
            Register the DEM processing subclass in the available methods

            :param dem_processing_subclass: the subclass to be registered
            :type dem_processing_subclass: object
            """
            cls.available_dem_processing_methods[dem_processing_method] = (
                dem_processing_subclass
            )
            return dem_processing_subclass

        return decorator
