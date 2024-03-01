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
import json
import logging
import os
from datetime import datetime
from importlib.metadata import version
from typing import List

# DEMcompare imports
from .internal_typing import ConfigType
from .sphinx_project_generator import SphinxProjectManager
from .stats_dataset import StatsDataset


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
            "pitch": (
                "This mode results relies only on **valid values** "
                "without nan values "
                "(whether they are "
                "from the error image or the reference support "
                "image when do_classification is on). Outliers and "
                "masked ones has been also discarded."
            )
        }
        modes_information[classification_layer_name]["intersection"] = {
            "pitch": (
                "This is the standard mode where only the pixels for "
                "which input DSMs classifications are intersection."
            )
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
                result = (
                    os.path.join(
                        working_dir,
                        "stats",
                        stats_dataset.dem_processing.type,
                        classification_layer_name,
                        "stats_results.csv",
                    ),
                )
            else:
                result = (
                    os.path.join(
                        working_dir,
                        "stats",
                        stats_dataset.dem_processing.type,
                        classification_layer_name,
                        f"stats_results_{mode}.csv",
                    ),
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
                                        (
                                            item
                                            if isinstance(item, str)
                                            else format(item, ".2f")
                                        )
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
    stats_dataset: StatsDataset,
    src: str,
) -> str:
    """
    Fill report with image views: snapshots, cdf, pdf information

    :param working_dir: directory in which to find
         *mode*.csv files for each mode in modename
    :param src: report source
    :return: filled src
    """

    dem_type = stats_dataset.dem_processing.type

    # for one dem for stats: snapshot, pdf, cdf
    dem_for_stats = os.path.join(
        working_dir,
        "stats",
        dem_type,
        "dem_for_stats_snapshot.png",
    )
    dem_for_stats_pdf = os.path.join(
        working_dir, "stats", dem_type, "dem_for_stats_pdf.png"
    )
    dem_for_stats_cdf = os.path.join(
        working_dir, "stats", dem_type, "dem_for_stats_cdf.png"
    )
    dem_for_stats_svf = os.path.join(
        working_dir, "stats", dem_type, "dem_for_stats_svf.png"
    )
    dem_for_stats_hillshade = os.path.join(
        working_dir,
        "stats",
        dem_type,
        "dem_for_stats_hillshade.png",
    )

    # -> show image snapshot, cdf, pdf without coregistration
    # TODO: add sampling source information
    src = "\n".join(
        [
            src,
            "",
            f"**{stats_dataset.dem_processing.fig_title}**",
            "==========================",
            "",
        ]
    )

    tmp = dem_type + "|"

    # if exists : -> one dem input REF
    if dem_for_stats:
        src = "\n".join(
            [
                src,
                "",
                "Image views",
                "===============",
                "",
                f".. |img_{dem_type}| image:: /{dem_for_stats}",
                "  :width: 100%",
                f".. |pdf_{dem_type}| image:: /{dem_for_stats_pdf}",
                "  :width: 100%",
                f".. |cdf_{dem_type}| image:: /{dem_for_stats_cdf}",
                "  :width: 100%",
                f".. |svf_{dem_type}| image:: /{dem_for_stats_svf}",
                "  :width: 100%",
                f".. |hillshade_{dem_type}| image:: /{dem_for_stats_hillshade}",
                "  :width: 100%",
                "",
                "+--------------------------------------"
                + "+--------------------------------------"
                + "+--------------------------------------+",
                "|   Image view                         "
                + "| Sky-view factor                      "
                + "| Hill shade                           |",
                "+--------------------------------------"
                + "+--------------------------------------"
                + "+--------------------------------------+",
                f"| |img_{tmp.ljust(23)}         "
                + f"| |svf_{tmp.ljust(22)}          "
                + f"|  |hillshade_{tmp.ljust(22)}   |",
                "+--------------------------------------"
                + "+--------------------------------------"
                + "+--------------------------------------+",
                "",
                "+--------------------------------------"
                + "+--------------------------------------+",
                "| Histogram                            "
                + "| Cumulative distribution              |",
                "+--------------------------------------"
                + "+--------------------------------------+",
                f"| |pdf_{tmp.ljust(23)}         "
                + f"|  |cdf_{tmp.ljust(28)}   |",
                "+--------------------------------------"
                + "+--------------------------------------+",
                "",
            ]
        )

    return src


def fill_report(
    cfg: ConfigType,
    stats_datasets: List[StatsDataset] = None,
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

    if stats_datasets:
        for stats_dataset in stats_datasets:
            # Show image views
            src = fill_report_image_views(working_dir, stats_dataset, src)
            # Fill statistics report
            if "statistics" in cfg:
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
    stats_datasets: List[StatsDataset] = None,
):
    """
    Generate demcompare report

    :param cfg: input demcompare configuration
    :param stats_dataset: stats dataset demcompare object containing results
    """

    # sphinx output documentation directory of demcompare report
    output_doc_dir = os.path.join(cfg["output_dir"], "report/published_report")
    # sphinx source documentation directory of demcompare report
    src_doc_dir = os.path.join(cfg["output_dir"], "report/src")

    # Initialize the sphinx project source and output build config
    spm = SphinxProjectManager(src_doc_dir, output_doc_dir, "index", "")

    # create report contents from demcompare cfg and stats datasets
    report_content = fill_report(cfg=cfg, stats_datasets=stats_datasets)

    # Add report contents to the sphinx project
    spm.write_body(report_content)

    # Build sphinx project in html and latex
    try:
        spm.build_project("html")
    except Exception:
        logging.error("Error when building report as html output (ignored)")
        raise
    try:
        spm.build_project("latexpdf")
    except Exception:
        # put only INFO to be more silent to logging when latexpdf not present
        # to change if we keep or not pdf report
        logging.info("Error when building report as pdf output (ignored)")

    # Sphinx project install in final directory
    spm.install_project()
