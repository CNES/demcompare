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
This module contains the Coregistration class factory.
This main API Coregistration class generates an object linked
with coregistration configuration "method_name"
(from registered CoregistrationTemplate class)
"""

# Standard imports
import logging
from typing import Any, Dict


class Coregistration:
    """
    Coregistration factory:
    A class designed for registered all available coregistration methods
    and instantiate them when needed.
    """

    # Dict (method_name: str, class: object) containing registered
    # available coregistration methods
    available_coregistrations: Dict[str, Any] = {}
    default_application = "nuth_kaab_internal"

    def __new__(cls, cfg: Dict[str, Any] = None):
        """
        Return a CoregistrationTemplate child instance
        associated with the "method_name" given in the configuration
        through create_coreg local method for clarity.

        :param cfg: JSON configuration {'method_name': value}
        :type cfg: Dict[str, Any]
        """
        return cls.create_coreg(cfg)

    @classmethod
    def create_coreg(cls, cfg: Dict[str, Any] = None):
        """
        Factory command to create the coregistration from method_name
        Return a CoregistrationTemplate child instance
        associated with the "method_name" given in the configuration

        :param cfg: configuration {'method_name': value}
        :type cfg: Dict[str, Any]
        """

        # If no cfg is given, use default_application
        coreg_method = cls.default_application
        if bool(cfg):
            coreg_method = cfg["method_name"]

        try:
            coreg_class = cls.available_coregistrations[coreg_method]
            coreg = coreg_class(cfg)
            logging.info("Coregistration method name: %s", coreg_method)

        except KeyError:
            logging.error(
                "No coregistration method named %s supported", coreg_method
            )
            raise

        return coreg

    @classmethod
    def print_avalaible_coregistration_methods(cls):
        """
        Print all registered applications
        """
        for coreg_method_name in cls.available_coregistrations:
            print(coreg_method_name)

    @classmethod
    def register(cls, coreg_method_name: str):
        """
        Allows to register the CoregistrationTemplate subclass in the factory
        with its coregistration method_name through decorator

        :param coreg_method_name: the subclass name to be registered
        :type coreg_method_name: string
        """

        def decorator(coreg_subclass):
            """
            Register the coregistration subclass in the available methods

            :param coreg_subclass: the subclass to be registered
            :type coreg_subclass: object
            """
            cls.available_coregistrations[coreg_method_name] = coreg_subclass
            return coreg_subclass

        return decorator
