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
# pylint:disable=too-many-locals, too-many-branches, broad-except, fixme
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
import glob
import json
import logging
import os
from datetime import datetime
from importlib.metadata import version
from typing import List

# DEMcompare imports
from .internal_typing import ConfigType
from .output_tree_design import get_out_dir
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
    Fill report with statistics information for all cases:
    without coreg:
    - two dems : diff of the two dems
    - one dsm : stats on the dem alone.

    with coreg and two dems: diff of two coregistered dems.

    TODO: if refacto report, use only csv or stats_dataset but not both

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
            "The `stats metrics <https://demcompare.readthedocs.io/"
            "en/stable/userguide/statistics.html#metrics>`_",
            "are organized around",
            "`classification layers and modes "
            "<https://demcompare.readthedocs.io/"
            "en/stable/userguide/statistics.html#classification-layers>`_.",
            "",
        ]
    )

    src = "\n".join(
        [
            src,
            "Stats are generated from:",
            "",
            "- **input_ref** when one input dem and statistics only",
            "- reprojected **input_ref - input_sec**"
            " difference when statistics only",
            "- reprojected and coregistered **input_ref - input_sec**"
            " difference with coregistration and statistics",
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


def fill_report_image_views(  # noqa: C901
    working_dir: str,
    src: str,
) -> str:
    """
    Fill report with image views: snapshots, cdf, pdf information

    :param working_dir: directory in which to find
         *mode*.csv files for each mode in modename
    :param src: report source
    :return: filled src
    """

    # TODO: replace with OTD name and not recursive search

    # Find snapshot files depending on mode (one ref dem or ref-sec diff)
    # for initial dem diff: snapshot, pdf, cdf
    initial_dem_diff = first_recursive_search(
        working_dir, "initial_dem_diff_snapshot.png"
    )
    initial_dem_diff_pdf = first_recursive_search(
        working_dir, "initial_dem_diff_pdf.png"
    )
    initial_dem_diff_cdf = first_recursive_search(
        working_dir, "initial_dem_diff_cdf.png"
    )

    # for final (after coreg) dem diff: snapshot, pdf, cdf
    final_dem_diff = first_recursive_search(
        working_dir, "final_dem_diff_snapshot.png"
    )
    final_dem_diff_pdf = first_recursive_search(
        working_dir, "final_dem_diff_pdf.png"
    )
    final_dem_diff_cdf = first_recursive_search(
        working_dir, "final_dem_diff_cdf.png"
    )

    # for one dem for stats: snapshot, pdf, cdf
    dem_for_stats = first_recursive_search(
        working_dir, "dem_for_stats_snapshot.png"
    )
    dem_for_stats_pdf = first_recursive_search(
        working_dir, "dem_for_stats_pdf.png"
    )
    dem_for_stats_cdf = first_recursive_search(
        working_dir, "dem_for_stats_cdf.png"
    )

    # -> show image snapshot, cdf, pdf without coregistration
    # TODO: add sampling source information
    src = "\n".join(
        [
            src,
            "",
            "Image views",
            "==========================",
            "",
        ]
    )
    # if exists : -> one dem input REF
    if dem_for_stats:
        src = "\n".join(
            [
                src,
                "",
                "**Initial elevation on one DEM (REF)**",
                "----------------------",
                "",
                f".. |img| image:: /{dem_for_stats}",
                "  :width: 100%",
                f".. |pdf| image:: /{dem_for_stats_pdf}",
                "  :width: 100%",
                f".. |cdf| image:: /{dem_for_stats_cdf}",
                "  :width: 100%",
                "",
                "+--------------+-------------+-------------------------+",
                "|   Image view | Histogram   | Cumulative distribution |",
                "+--------------+-------------+-------------------------+",
                "| |img|        | |pdf|       |  |cdf|                  |",
                "+--------------+-------------+-------------------------+",
                "",
            ]
        )

    # if exists : -> two dem initial diff with or without coregistration
    if initial_dem_diff:
        src = "\n".join(
            [
                src,
                "",
                "**Initial elevation (REF-SEC)**",
                "----------------------",
                "",
                f".. |img| image:: /{initial_dem_diff}",
                "  :width: 100%",
                f".. |pdf| image:: /{initial_dem_diff_pdf}",
                "  :width: 100%",
                f".. |cdf| image:: /{initial_dem_diff_cdf}",
                "  :width: 100%",
                "",
                "+--------------+-------------+-------------------------+",
                "|   Image view | Histogram   | Cumulative distribution |",
                "+--------------+-------------+-------------------------+",
                "| |img|        | |pdf|       |  |cdf|                  |",
                "+--------------+-------------+-------------------------+",
                "",
            ]
        )

    # if exists : ->  differences with coregistration
    if final_dem_diff:
        src = "\n".join(
            [
                src,
                "**Final elevation after coregistration"
                " (COREG_REF-COREG_SEC)**",
                "-----------------------------------------",
                "",
                f".. |img2| image:: /{final_dem_diff}",
                "  :width: 100%",
                f".. |pdf2| image:: /{final_dem_diff_pdf}",
                "  :width: 100%",
                f".. |cdf2| image:: /{final_dem_diff_cdf}",
                "  :width: 100%",
                "",
                "+--------------+-------------+-------------------------+",
                "|   Image view | Histogram   | Cumulative distribution |",
                "+--------------+-------------+-------------------------+",
                "| |img2|       | |pdf2|      |  |cdf2|                 |",
                "+--------------+-------------+-------------------------+",
                "",
                "",
            ]
        )

    return src


def fill_report(
    cfg: ConfigType,
    stats_dataset: StatsDataset = None,
) -> str:
    """
    Fill sphinx demcompare report into a string from cfg and stats_dataset

    :param cfg: input demcompare configuration
    :param stats_dataset: stats dataset demcompare object containing results
    """
    # output directory in which to find demcompare outputs info for report
    working_dir = cfg["output_dir"]

    # Create source sphinx for demcompare report

    # -> header part
    now = datetime.now()
    date = now.strftime("%d/%m/%y %Hh%M")
    version_ = version("demcompare")
    src = "\n".join(
        [
            "",
            "*********************",
            " Demcompare report   ",
            "*********************",
            "",
            f"- *Generated date:* {date}",
            f"- *Demcompare version:* {version_}",
            "- *Documentation:*"
            " `https://demcompare.readthedocs.io/ "
            "<https://demcompare.readthedocs.io/>`_",
            "- *Repository:*"
            " `https://github.com/CNES/demcompare "
            "<https://github.com/CNES/demcompare>`_",
            "",
        ]
    )

    # Show image views (dependent on initial, final or dem_for_stats images)
    src = fill_report_image_views(working_dir, src)

    # Fill statistics report
    if "statistics" in cfg and stats_dataset:
        src = fill_report_stats(working_dir, stats_dataset, src)

    # Show full input configuration with indent last line manually (bug)
    show_cfg = json.dumps(cfg, indent=2)
    show_cfg = show_cfg[:-1] + "   }"
    src = "\n".join(
        [
            src,
            "Full input configuration",
            "==========================",
            "",
            ".. code-block:: json",
            "",
            f"   {show_cfg}",
            "",
            "",
        ]
    )
    return src


def generate_report(  # noqa: C901
    cfg: ConfigType,
    stats_dataset: StatsDataset = None,
):
    """
    Generate demcompare report

    :param cfg: input demcompare configuration
    :param stats_dataset: stats dataset demcompare object containing results
    """

    # sphinx output documention directory of demcompare report
    output_doc_dir = os.path.join(
        cfg["output_dir"], get_out_dir("sphinx_built_doc")
    )
    # sphinx source documentation directory for report from OTD
    src_doc_dir = os.path.join(cfg["output_dir"], get_out_dir("sphinx_src_doc"))

    # Initialize the sphinx project source and output build config
    spm = SphinxProjectManager(src_doc_dir, output_doc_dir, "index", "")

    # create source

    src = fill_report(cfg=cfg, stats_dataset=stats_dataset)

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
