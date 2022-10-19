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
# pylint:disable=too-many-locals, too-many-branches, broad-except
"""
Generate demcompare report from DEM comparison results (graphs, stats, ...)

Steps:
- init SphinxProjectManager class structure
- create all data and documentation structure for each classif_layer and mode
- create sphinx source rst report and add to SphinxProjectManager object
- Compile SphinxProjectManager object to generate html and pdf report
"""

# Standard imports
import collections
import csv
import fnmatch
import glob
import logging
import os
import sys
from typing import List

# DEMcompare imports
from .sphinx_project_generator import SphinxProjectManager
from .stats_dataset import StatsDataset


def recursive_search(directory: str, pattern: str) -> List[str]:
    """
    Recursively look up pattern filename into dir tree

    :param directory: directory
    :type directory: str
    :param pattern: pattern
    :type pattern: str
    :return: search matches
    :rtype: List[str]
    """

    if sys.version[0:3] < "3.5":
        matches = []
        for root, _, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(root, filename))
    else:
        matches = glob.glob(f"{directory}/**/{pattern}", recursive=True)

    return matches


def first_recursive_search(directory: str, pattern: str):
    """
    Recursively look up pattern filename into dir tree
    with first found result
    if no results return None

    :param directory: directory
    :type directory: str
    :param pattern: pattern
    :type pattern: str
    :return: None
    """
    result = recursive_search(directory, pattern)
    if len(result) > 0:
        return result[0]
    # else
    return None


def fill_report_stats(
    working_dir: str, stats_dataset: StatsDataset, src: str
) -> str:
    """
    Fill report with statistics information

    :param working_dir: directory in which to find
         *mode*.csv files for each mode in modename
    :type working_dir: str
    :param stats_dataset: StatsDataset object
    :type stats_dataset: StatsDataset
    :param src: report source
    :type src: str
    :return: filled src
    :rtype:
    """

    classification_layer_masks = list(
        stats_dataset.classif_layers_and_modes.keys()
    )

    # Initialize mode informations
    modes_information: collections.OrderedDict = collections.OrderedDict()

    # Loop on demcompare classification_layer_masks
    for classification_layer_name in classification_layer_masks:
        # Initialize mode informations for classification_layer
        modes_information[classification_layer_name] = collections.OrderedDict()
        modes_information[classification_layer_name]["standard"] = {
            "pitch": "This mode results relies only on **valid values** "
            "without nan values "
            "(whether they are from the error image or the reference support "
            "image when do_classification is on). Outliers and "
            "masked ones has been also discarded."
        }
        modes_information[classification_layer_name]["intersection"] = {
            "pitch": "This is the standard mode where only the pixels for "
            "which input DSMs classifications are intersection."
        }
        modes_information[classification_layer_name]["exclusion"] = {
            "pitch": "This mode is the 'intersection' " "complementary."
        }
        modes = [
            "standard",
            "intersection",
            "exclusion",
        ]

        # Loop on demcompare modes.
        for mode in modes:
            # for mode in modes_information:
            # find csv stats associated with the mode
            # - csv
            if mode == "standard":
                result = recursive_search(
                    os.path.join(working_dir, "*", classification_layer_name),
                    "*.csv",
                )
            else:
                result = recursive_search(
                    os.path.join(working_dir, "*", classification_layer_name),
                    f"*_{mode}*.csv",
                )
            if len(result) > 0:
                if os.path.exists(result[0]):
                    csv_data = []
                    with open(result[0], "r", encoding="utf8") as csv_file:
                        csv_lines_reader = csv.reader(
                            csv_file, quoting=csv.QUOTE_NONNUMERIC
                        )
                        for row in csv_lines_reader:
                            csv_data.append(
                                ",".join(
                                    [
                                        item
                                        if isinstance(item, str)
                                        else format(item, ".2f")
                                        for item in row
                                    ]
                                )
                            )
                    modes_information[classification_layer_name][mode][
                        "csv"
                    ] = "\n".join(
                        [
                            "    " + csv_single_data
                            for csv_single_data in csv_data
                        ]
                    )
                else:
                    modes_information[classification_layer_name][mode][
                        "csv"
                    ] = None
            else:
                modes_information[classification_layer_name][mode]["csv"] = None
        # End of mode loop
    # End of classification_layer loop

    # -> stats results table of contents
    src = "\n".join([src, "Stats Results", "===============", ""])
    src = "\n".join(
        [
            src,
            "*Important: stats are generated on stats DEM *",
            "",
        ]
    )

    src = "\n".join(
        [
            src,
            "The stats are organized around "
            + "classification layers and modes.\n",
            "See `README Documentation "
            "<https://github.com/CNES/demcompare>`_ for details.",
            "\n",
            "**Classification layers:**",
            "",
        ]
    )
    for (
        classification_layer_name,
        modes_dict,
    ) in stats_dataset.classif_layers_and_modes.items():
        src = "\n".join(
            [
                src,
                "* The :ref:`{classification_layer_name}"
                " <{classification_layer_name}>` ".format(
                    classification_layer_name=classification_layer_name
                )
                + "classification layer"
                "",
            ]
        )

        src = "\n".join(
            [
                src,
                "**Evaluation modes:**",
                "",
            ]
        )
        for mode in modes_dict["modes"]:
            the_mode_pitch = modes_information[classification_layer_name][mode][
                "pitch"
            ]
            src = "\n".join(
                [
                    src,
                    "* The :ref:`{mode} <{mode}>` mode".format(mode=mode),
                    "",
                    f"{the_mode_pitch} \n",
                    "",
                ]
            )

    # -> the results
    for (
        classification_layer_name,
        modes_dict,
    ) in stats_dataset.classif_layers_and_modes.items():
        lines = "-" * len(classification_layer_name)
        src = "\n".join(
            [
                src,
                f".. _{classification_layer_name}:",
                "",
                f"Classification layer: {classification_layer_name}",
                f"{lines}-----------------------",
                "",
            ]
        )
        # loop for results for each mode
        for mode in modes_dict["modes"]:
            the_mode_csv = modes_information[classification_layer_name][mode][
                "csv"
            ]
            lines = "^" * len(mode)
            src = "\n".join(
                [
                    src,
                    "",
                    f".. _{mode}:",
                    "",
                    f"Mode: {mode}",
                    f"^^^^^^^^^^^^^^^{lines}",
                    "",
                ]
            )
            # Table of results
            if the_mode_csv:
                src = "\n".join(
                    [
                        src,
                        "Table showing comparison metrics",
                        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
                        ".. csv-table::",
                        "",
                        f"{the_mode_csv}",
                        "",
                    ]
                )
    return src


def fill_report_coreg(  # noqa: C901
    working_dir: str,
    sec_name: str,
    ref_name: str,
    coreg_sec_name: str,
    coreg_ref_name: str,
    src: str,
) -> str:
    """
    Fill report with coregistration information

    :param working_dir: directory in which to find
         *mode*.csv files for each mode in modename
    :type working_dir: str
    :param sec_name: name or path to the
      sec to be compared against the ref
    :type sec_name: str
    :param ref_name: name or path to the reference sec
    :type ref_name: str
    :param coreg_sec_name: name or path to the coreg sec
    :type coreg_sec_name: str
    :param coreg_ref_name: name or path to the ref sec
    :type coreg_ref_name: str
    :param src: report source
    :type src: str
    :return: filled src
    :rtype: str
    """

    # Find DSMs differences files
    dem_diff_without_coreg = first_recursive_search(
        working_dir, "initial_dem_diff.png"
    )
    dem_diff_with_coreg = first_recursive_search(
        working_dir, "final_dem_diff.png"
    )

    # Find DSMs CDF differences
    dem_diff_cdf_without_coreg = first_recursive_search(
        working_dir, "initial_dem_diff_cdf.png"
    )
    dem_diff_cdf_with_coreg = first_recursive_search(
        working_dir, "final_dem_diff_cdf.png"
    )

    # Find histogram files
    initial_dem_diff_pdf = first_recursive_search(
        working_dir, "initial_dem_diff_pdf.png"
    )
    final_dem_diff_pdf = first_recursive_search(
        working_dir, "final_dem_diff_pdf.png"
    )
    # Get ref_name
    sec_name_dir, sec_name = os.path.split(sec_name)
    ref_name_dir, ref_name = os.path.split(ref_name)
    coreg_sec_name_dir, coreg_sec_name = os.path.split(coreg_sec_name)
    coreg_ref_name_dir, coreg_ref_name = os.path.split(coreg_ref_name)

    # -> DSM differences without coregistration
    src = "\n".join(
        [
            src,
            "",
            "Elevation differences",
            "==========================",
            "",
            "**Without coregistration**",
            "--------------------------",
            f".. image:: /{dem_diff_without_coreg}",
            f".. image:: /{dem_diff_cdf_without_coreg}",
            "",
            "*Input Initial DEMs:*",
            "",
            f"* Tested DEM (SEC): {sec_name}",
            f"   * dir: {sec_name_dir}",
            f"* Reference DEM (REF): {ref_name}",
            f"   * dir: {ref_name_dir}",
            "",
        ]
    )
    # -> Elevation Difference Histogram without coregistration
    src = "\n".join(
        [
            src,
            "**Elevation difference histogram on all pixels"
            + " without coregistration**",
            "-----------------------",
            f".. image:: /{initial_dem_diff_pdf}",
            "",
        ]
    )

    # if exists : -> DSM differences with coregistration
    if dem_diff_with_coreg:
        src = "\n".join(
            [
                src,
                "**With coregistration**",
                "-----------------------",
                f".. image:: /{dem_diff_with_coreg}",
                f".. image:: /{dem_diff_cdf_with_coreg}",
                "",
            ]
        )
        src = "\n".join(
            [
                src,
                "**Generated coregistered DEMs:**",
                "",
                f"* Tested Coreg DEM (COREG_SEC): {coreg_sec_name}",
                f"   * dir: {coreg_sec_name_dir} ",
                f"* Reference Coreg DEM (COREG_REF): {coreg_ref_name}",
                f"   * dir: {coreg_ref_name_dir}",
                "",
            ]
        )

        # -> Elevation Difference Histogram with coregistration
        src = "\n".join(
            [
                src,
                "**Elevation difference histogram on all pixels "
                + "with coregistration**",
                "-----------------------",
                f".. image:: /{final_dem_diff_pdf}",
                "",
            ]
        )
    return src


def generate_report(  # noqa: C901
    working_dir: str,
    stats_dataset: StatsDataset = None,
    sec_name: str = None,
    ref_name: str = None,
    coreg_sec_name: str = None,
    coreg_ref_name: str = None,
    doc_dir: str = ".",
    project_dir: str = ".",
):
    """
    Generate demcompare report

    :param working_dir: directory in which to find
         *mode*.csv files for each mode in modename
    :type working_dir: str
    :param sec_name: name or path to the
      sec to be compared against the ref
    :type sec_name: str
    :param ref_name: name or path to the reference sec
    :type ref_name: str
    :param coreg_sec_name: name or path to the coreg sec
    :type coreg_sec_name: str
    :param coreg_ref_name: name or path to the ref sec
    :type coreg_ref_name: str
    :param src: report source
    :type src: str
    :return: filled src
    :rtype: str
    """
    # Initialize the sphinx project
    spm = SphinxProjectManager(
        project_dir, doc_dir, "demcompare_report", "DEM Compare Report"
    )

    # Create source
    # -> header part
    src = "\n".join(
        [
            ".. _DEM_COMPARE_REPORT:",
            "",
            "*********************",
            " Demcompare Report ",
            "*********************" "",
            "This report is generated by "
            "`DEMCompare <https://github.com/CNES/demcompare>`_",
            "",
        ]
    )

    if sec_name and ref_name and coreg_ref_name and coreg_sec_name:
        src = fill_report_coreg(
            working_dir, sec_name, ref_name, coreg_sec_name, coreg_ref_name, src
        )

    if stats_dataset:
        src = fill_report_stats(working_dir, stats_dataset, src)

    build_report(spm, src)


def build_report(spm: SphinxProjectManager, src: str) -> None:
    """
    Build the demcompare report

    :param src: report source
    :type src: str
    :param spm: SphinxProjectManager
    :type spm: SphinxProjectManager
    :return: None
    """

    # Add source to the project
    spm.write_body(src)

    # Build the project
    try:
        spm.build_project("html")
    except Exception:
        logging.error("Error when building report as html output (ignored)")
        raise
    try:
        spm.build_project("latexpdf")
    except Exception:
        logging.error("Error when building report as pdf output (ignored)")

    # Sphinx project install
    spm.install_project()
