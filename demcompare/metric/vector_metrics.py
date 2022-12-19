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
Mainly contains different 2D metric classes
"""

import csv
import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as mpl_pyplot
import numpy as np
from astropy import units as u

from .metric import Metric
from .metric_template import MetricTemplate


@Metric.register("cdf")
class CumulativeProbabilityFunction(MetricTemplate):
    """
    Cumulative Probability Function metric class
    """

    # Default bin step for histogram computation
    _BIN_STEP = 0.1

    def __init__(self, parameters: Dict = None):
        """
        Initialization the metric object

        :param parameters: optional input parameters
        :type parameters: dict
        :return: None
        """

        super().__init__(parameters)
        # Metric type
        self.type = "vector"
        # Plot attributes
        self.nb_bins: int = None
        self.max_diff: float = None
        self.nb_nans: int = None
        self.nb_pixels: int = None
        self.bins_count: np.ndarray = None
        self.cdf: np.ndarray = None
        self.output_csv_path: str = None
        self.output_plot_path: str = None
        self.bin_step: float = self._BIN_STEP

        # Bin step
        if parameters:
            if "bin_step" in parameters:
                self.bin_step = parameters["bin_step"]
            if "output_csv_path" in parameters:
                self.output_csv_path = parameters["output_csv_path"]
            if "output_plot_path" in parameters:
                self.output_plot_path = parameters["output_plot_path"]

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed cdf (y axis) and bins (y axis)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Generate absolute values array
        data = np.abs(data)
        # Get max diff from data
        self.max_diff = np.nanmax(data)
        # Count nb nan
        self.nb_nans = np.sum(np.isnan(data))
        self.nb_pixels = data.shape
        # Get bins number for histogram
        self.nb_bins = int(self.max_diff / self.bin_step)
        # getting data of the histogram
        hist, self.bins_count = np.histogram(
            data,
            bins=self.nb_bins,
            range=(0, self.max_diff),
            density=True,
        )
        # Normalized Probability Density Function of the histogram
        pdf = hist / sum(hist)
        # Generate Cumulative Probability Function
        self.cdf = np.cumsum(pdf)

        if self.output_csv_path:
            self.save_csv_metric(self.output_csv_path)
        if self.output_plot_path:
            self.save_plot_metric(self.output_plot_path)
        return self.cdf, self.bins_count

    def save_csv_metric(self, output_file: str):
        """
        Save the metric to a csv file

        :param output_file: path where the csv file is saved
        :type output_file: str
        :return: None
        """
        # Get plot_file file base
        plot_file_base = os.path.splitext(output_file)[0]

        # Save cdf in csv in same base file name.
        with open(
            plot_file_base + ".csv", "w", newline="", encoding="utf8"
        ) as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["Bins", "CDF values"])
            writer.writerows(zip(self.bins_count, self.cdf))  # noqa: B905

    def save_plot_metric(self, output_file: str):
        """
        Compute and save the metric plot

        :param output_file: path where the plot image is saved
        :type output_file: str
        :return: None
        """
        # Plot
        fig, fig_ax = mpl_pyplot.subplots()
        fig_ax.plot(self.bins_count[1:], self.cdf, label="CDF")
        # tidy up the figure and add axes titles
        fig_ax.set_xlabel(
            "Full absolute elevation differences (m) "
            f"\nmax_diff={round(self.max_diff, 3)} nb_bins={self.nb_bins}"
            f"\nnb_pixels={self.nb_pixels} nb_nans={self.nb_nans}",
            fontsize="medium",
        )
        fig_ax.set_ylabel("Cumulative Probability [0,1]", fontsize="medium")
        fig_ax.set_title("Cummulative Probability Difference", fontsize="large")

        fig_ax.set_ylim(0, 1.05)
        fig_ax.grid(True)

        fig.savefig(output_file, dpi=100, bbox_inches="tight")
        mpl_pyplot.close()


@Metric.register("pdf")
class ProbabilityDensityFunction(MetricTemplate):
    """
    Probability Density Function metric class
    """

    # Default bin step for histogram computation
    _BIN_STEP = 0.2
    _WIDTH = 0.7

    def __init__(self, parameters: Dict = None):
        """
        Initialization the metric object

        :param parameters: optional input parameters
        :type parameters: dict
        :return: None
        """

        super().__init__(parameters)
        # Metric type
        self.type = "vector"
        # Plot attributes
        self.bin_step: float = None
        self.width: float = None
        self.filter_p98: float = None
        self.pdf: np.ndarray = None
        self.bins: np.ndarray = None
        self.output_csv_path: str = None
        self.output_plot_path: str = None

        if parameters:
            if "bin_step" in parameters:
                self.bin_step = parameters["bin_step"]
            else:
                self.bin_step = self._BIN_STEP
            if "width" in parameters:
                self.width = parameters["width"]
            else:
                self.width = self._WIDTH
            if "filter_p98" in parameters:
                self.filter_p98 = parameters["filter_p98"] == "True"
            else:
                self.filter_p98 = False
            if "output_csv_path" in parameters:
                self.output_csv_path = parameters["output_csv_path"]
            if "output_plot_path" in parameters:
                self.output_plot_path = parameters["output_plot_path"]

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed pdf (y axis) and bins (y axis)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """

        # Histogram plot creation.
        if self.filter_p98:
            # The histogram is centered around 0
            # and bounded over [- |percentile98|, |percentile98|]
            bound = np.abs(np.nanpercentile(data, 98))
        else:
            bound = np.nanmax(data)

        hist, self.bins = np.histogram(
            data[~np.isnan(data)],
            bins=np.arange(-bound, bound, self.bin_step),
        )

        # Normalized Probability Density Function of the histogram
        self.pdf = hist / sum(hist)

        if self.output_csv_path:
            self.save_csv_metric(self.output_csv_path)
        if self.output_plot_path:
            self.save_plot_metric(self.output_plot_path)
        return self.pdf, self.bins

    def save_csv_metric(self, output_file: str):
        """
        Save the metric to a csv file

        :param output_file: path where the csv file is saved
        :type output_file: str
        :return: None
        """
        # Get plot_file file base
        plot_file_base = os.path.splitext(output_file)[0]

        # Save pdf in csv in same base file name.
        with open(
            plot_file_base + ".csv", "w", newline="", encoding="utf8"
        ) as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["Bins", "Normalized PDF values"])
            writer.writerows(zip(self.bins, self.pdf))  # noqa: B905

    def save_plot_metric(self, output_file: str):
        """
        Compute and save the metric plot

        :param output_file: path where the plot image is saved
        :type output_file: str
        :return: None
        """
        # Define histogram's bins width
        width = self.width * (self.bins[1] - self.bins[0])
        center = (self.bins[:-1] + self.bins[1:]) / 2
        fig0 = mpl_pyplot.figure()
        # Define axes labels and title
        fig0_ax = fig0.add_subplot(111)
        fig0_ax.set_xlabel(
            "Elevation difference (m) from - |p98| to |p98|", fontsize=12
        )
        fig0_ax.set_title("Elevation difference Histogram", fontsize="large")
        fig0_ax.set_ylabel("Normalized frequency", fontsize=12)
        mpl_pyplot.grid(True)
        mpl_pyplot.bar(center, self.pdf, align="center", width=width)
        mpl_pyplot.savefig(
            output_file,
            dpi=100,
            bbox_inches="tight",
        )


@Metric.register("ratio_above_threshold")
class RatioAboveThreshold(MetricTemplate):
    """
    Ratio above threshold metric class
    """

    # Default elevation thresholds in meters
    _ELEVATION_THRESHOLDS = [0.5, 1, 3]
    _ORIGINAL_UNIT = "m"

    def __init__(self, parameters: Dict = None):
        """
        Initialization the metric object

        :param parameters: optional input parameters
        :type parameters: dict
        :return: None
        """

        super().__init__(parameters)
        # Metric type
        self.type = "vector"
        self.ratio_above_thrshld: List = None
        self.output_csv_path: str = None
        # Elevation thresholds
        if parameters:
            threshold = parameters["elevation_threshold"]
            if "original_unit" in parameters:
                original_unit = parameters["original_unit"]
            else:
                original_unit = self._ORIGINAL_UNIT
            self.elevation_threshold = self._get_thresholds_in_meters(
                threshold, original_unit
            )
            if "output_csv_path" in parameters:
                self.output_csv_path = parameters["output_csv_path"]
        else:
            self.elevation_threshold = self._ELEVATION_THRESHOLDS

    @staticmethod
    def _get_thresholds_in_meters(threshold: List[float], original_unit: str):
        """
        Create list of threshold in meters.
        """
        # Convert thresholds to meter
        # since demcompare's elevation unit is "meter"

        list_threshold_m = [
            ((threshold * u.Unit(original_unit)).to(u.meter)).value
            for threshold in threshold
        ]
        return list_threshold_m

    def compute_metric(
        self, data: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Metric computation method

        :param data: input data to compute the metric
        :type data: np.array
        :return: the computed ratio_above_threshold
        :rtype: np.ndarray
        """
        self.ratio_above_thrshld = []
        for threshold in self.elevation_threshold:
            self.ratio_above_thrshld.append(
                (np.count_nonzero(data > threshold)) / float(data.size),
            )
        if self.output_csv_path:
            self.save_csv_metric(self.output_csv_path)
        return np.array(self.ratio_above_thrshld), np.array(
            self.elevation_threshold
        )

    def save_csv_metric(self, output_file: str):
        """
        Save the metric to a csv file

        :param output_file: path where the csv file is saved
        :type output_file: str
        :return: None
        """
        # Get plot_file file base
        plot_file_base = os.path.splitext(output_file)[0]

        # Save pdf in csv in same base file name.
        with open(
            plot_file_base + ".csv", "w", newline="", encoding="utf8"
        ) as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["Thresholds", "Ratio above threshold"])
            writer.writerows(
                zip(  # noqa: B905
                    self.elevation_threshold, self.ratio_above_thrshld
                )
            )
