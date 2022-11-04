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
Mainly contains the Metric class.
A metric defines a way to compute a statistic on an input data
such as an altitude difference map
"""

import logging
from typing import Any, Dict, Union


class Metric:
    """
    Metric factory:
    A class designed for registered all available metric methods
    and instantiate them when needed.
    """

    available_metrics: Dict[str, Any] = {}

    def __new__(cls, metric_method: str, params: Union[Dict, None] = None):
        """
        Return a MetricTemplate child instance
        associated with the "metric_method" given in the configuration
        through metric_method local method for clarity.

        :param metric_method: metric method
        :type metric_method: str
        """
        return cls.create_metric(metric_method, params)

    @classmethod
    def create_metric(cls, metric_method: str, parameters: Dict = None):
        """
        Factory command to create the metric from method_name
        Return a MetricTemplate child instance
        associated with the name given in the configuration

        :param metric_method: metric method
        :type metric_method: str
        :param parameters: optional input parameters
        :type parameters: Dict
        """

        try:
            metric_class = cls.available_metrics[metric_method]
            metric = metric_class(parameters=parameters)
        except KeyError:
            logging.error("No metric layer type %s supported", metric_method)
            raise
        return metric

    @classmethod
    def print_avalaible_metric_methods(cls):
        """
        Print all registered metric methods
        """
        for metric_method in cls.available_metrics:
            print(metric_method)

    @classmethod
    def register(cls, metric_method: str):
        """
        Allows to register the MetricTemplate
        subclass in the factory
        with its metric type through decorator

        :param metric_method: the subclass type to be registered
        :type metric_method: string
        """

        def decorator(metric_subclass):
            """
            Register the metric subclass in the available methods

            :param metric_subclass: the subclass to be registered
            :type metric_subclass: object
            """
            cls.available_metrics[metric_method] = metric_subclass
            return metric_subclass

        return decorator
