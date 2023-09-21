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
Mainly contains the MetricTemplate class.
"""
# Standard imports
from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Union

# Third party imports
import numpy as np


class MetricTemplate(
    metaclass=ABCMeta
):  # pylint:disable=too-few-public-methods
    """
    Metric class
    """

    # Default metric type
    DEFAULT_TYPE = "scalar"

    def __init__(
        self, parameters: Dict = None
    ):  # pylint:disable = unused-argument
        """
        Initialization of a metric object

        :param parameters: optional input parameters
        :type parameters: str
        :return: None
        """

        # Metric type
        self.type = self.DEFAULT_TYPE

    @abstractmethod
    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method
        """
