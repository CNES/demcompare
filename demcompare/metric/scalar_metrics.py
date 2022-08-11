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
Mainly contains different scalar metric classes
"""
from typing import Tuple, Union

import numpy as np

from .metric import Metric
from .metric_template import MetricTemplate


@Metric.register("mean")
class Mean(MetricTemplate):
    """
    Mean metric class
    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed mean
        :rtype: float
        """
        mean = np.nanmean(data)
        return mean


@Metric.register("max")
class Max(MetricTemplate):
    """
    Max metric class
    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed max
        :rtype: float
        """
        computed_max = np.max(data)
        return computed_max


@Metric.register("min")
class Min(MetricTemplate):
    """
    Min metric class
    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed min
        :rtype: float
        """
        computed_min = np.min(data)
        return computed_min


@Metric.register("std")
class Std(MetricTemplate):
    """
    Standard deviation metric class

    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed std
        :rtype: float
        """
        std = np.std(data)
        return std


@Metric.register("rmse")
class Rmse(MetricTemplate):
    """
    Root-mean-square-deviation metric class

    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed rmse
        :rtype: rmse
        """
        rmse = np.sqrt(np.mean(data * data))
        return rmse


@Metric.register("median")
class Median(MetricTemplate):
    """
    Median metric class
    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed median
        :rtype: float
        """
        median = np.nanmedian(data)
        return median


@Metric.register("nmad")
class Nmad(MetricTemplate):
    """
    Normalized-median-absolute-deviation metric class
    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed nmad
        :rtype: float
        """
        nmad = 1.4826 * np.nanmedian(np.abs(data - np.nanmedian(data)))
        return nmad


@Metric.register("sum")
class Sum(MetricTemplate):
    """
    Summation metric class
    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed sum
        :rtype: float
        """
        computed_sum = np.sum(data)
        return computed_sum


@Metric.register("squared_sum")
class SumSquaredErr(MetricTemplate):
    """
    Squared summation metric class
    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed squared_sum
        :rtype: float
        """
        squared_sum = np.sum(data * data)
        return squared_sum


@Metric.register("percentil_90")
class Percentil90(MetricTemplate):
    """
    90 percentil metric class
    """

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed percentil_90
        :rtype: float
        """
        p_90 = np.nanpercentile(np.abs(data - np.nanmean(data)), 90)
        return p_90
