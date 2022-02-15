#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2021 Centre National d'Etudes Spatiales (CNES).
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
Generate demcompare report from DEM comparison results (graphs, stats, ...)

Steps:
- init SphinxProjectManager class structure
- create all data and documentation needed structure for each partition and mode
- create sphinx source rst report and add to SphinxProjectManager object
- Compile SphinxProjectManager object to generate html and pdf report

TODO: if this report part is kept "as is", split generate_report in subfunctions
"""

# Standard imports
import collections
import csv
import fnmatch
import glob
import os
import sys
from typing import List

# DEMcompare imports
from .sphinx_project_generator import SphinxProjectManager


def recursive_search(directory: str, pattern: str):
    """
    Recursively look up pattern filename into dir tree

    :param directory:
    :param pattern:
    :return: search matches
    """ ""

    if sys.version[0:3] < "3.5":
        matches = []
        for root, _, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(root, filename))
    else:
        matches = glob.glob(
            "{}/**/{}".format(directory, pattern), recursive=True
        )

    return matches


def first_recursive_search(directory: str, pattern: str):
    """
    Recursively look up pattern filename into dir tree
    with first found result
    if no results return None

    :param directory:
    :type directory: str
    :param pattern:
    :type pattern: str
    :return: matches_number matches result
    """
    result = recursive_search(directory, pattern)
    if len(result) > 0:
        return result[0]
    # else
    return None


def generate_report(  # noqa: C901
    working_dir: str,
    dsm_name: str,
    ref_name: str,
    coreg_dsm_name: str,
    coreg_ref_name: str,
    partitions: List[str] = None,
    doc_dir: str = ".",
    project_dir: str = ".",
):
    """
    Create pdf and html report from png graph and csv stats summary

    :param working_dir: directory in which to find *mode*.png
            and *mode*.csv files for each mode in modename
    :type working_dir: str
    :param dsm_name: name or path to the dsm to be compared against the ref
    :type dsm_name: str
    :param ref_name: name or path to the reference dsm
    :type ref_name: str
    :param coreg_dsm_name: name or path to the coreg dsm
    :type coreg_dsm_name: str
    :param coreg_ref_name: name or path to the ref dsm
    :type coreg_ref_name: str
    :param partitions: list of partition, contains modes by partition
    :type partitions: List[str]
    :param doc_dir: directory in which to find the output documentation
    :type doc_dir: str
    :param project_dir: directory of the sphinx src documentation
    :type project_dir: str
    :return:
    """

    if partitions is None:
        partitions = ["standard"]

    # Initialize the sphinx project
    spm = SphinxProjectManager(
        project_dir, doc_dir, "demcompare_report", "DEM Compare Report"
    )

    # TODO modes_information[mode] overwritten , needs one per partition
    # => modes_information[partition_name][mode]

    # Initialize mode informations
    modes_information = collections.OrderedDict()

    # Loop on demcompare partitions
    for partition_name, _ in partitions.items():
        # Initialize mode informations for partition
        modes_information[partition_name] = collections.OrderedDict()
        modes_information[partition_name]["standard"] = {
            "pitch": "This mode results relies only on **valid values** "
            "without nan values "
            "(whether they are from the error image or the reference support "
            "image when do_classification is on). Outliers and "
            "masked ones has been also discarded."
        }
        modes_information[partition_name]["coherent-classification"] = {
            "pitch": "This is the standard mode where only the pixels for "
            "which input DSMs classifications are coherent."
        }
        modes_information[partition_name]["incoherent-classification"] = {
            "pitch": "This mode is the 'coherent-classification' "
            "complementary."
        }
        modes = [
            "standard",
            "coherent-classification",
            "incoherent-classification",
        ]

        # Loop on demcompare modes.
        for mode in modes:
            # for mode in modes_information:
            # find both graph and csv stats associated with the mode
            # - histograms
            result = recursive_search(
                os.path.join(working_dir, "*", partition_name),
                "*Real*_{}*.png".format(mode),
            )
            if len(result) > 0:
                modes_information[partition_name][mode]["histo"] = result[0]
            else:
                modes_information[partition_name][mode]["histo"] = None

            # - graph
            result = recursive_search(
                os.path.join(working_dir, "*", partition_name),
                "*Fitted*_{}*.png".format(mode),
            )
            if len(result) > 0:
                modes_information[partition_name][mode][
                    "fitted_histo"
                ] = result[0]
            else:
                modes_information[partition_name][mode]["fitted_histo"] = None
            # - csv
            result = recursive_search(
                os.path.join(working_dir, "*", partition_name),
                "*_{}*.csv".format(mode),
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
                    modes_information[partition_name][mode]["csv"] = "\n".join(
                        [
                            "    " + csv_single_data
                            for csv_single_data in csv_data
                        ]
                    )
                else:
                    modes_information[partition_name][mode]["csv"] = None
            else:
                modes_information[partition_name][mode]["csv"] = None
        # End of mode loop
    # End of partition loop

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
    dsm_name_dir, dsm_name = os.path.split(dsm_name)
    ref_name_dir, ref_name = os.path.split(ref_name)
    if dem_diff_with_coreg:
        coreg_dsm_name_dir, coreg_dsm_name = os.path.split(coreg_dsm_name)
        coreg_ref_name_dir, coreg_ref_name = os.path.split(coreg_ref_name)

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
            ".. image:: /{}".format(dem_diff_without_coreg),
            ".. image:: /{}".format(dem_diff_cdf_without_coreg),
            "",
            "*Input Initial DEMs:*",
            "",
            "* Tested DEM (DEM): {}".format(dsm_name),
            "   * dir: {}".format(dsm_name_dir),
            "* Reference DEM (REF): {}".format(ref_name),
            "   * dir: {}".format(ref_name_dir),
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
            ".. image:: /{}".format(initial_dem_diff_pdf),
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
                ".. image:: /{}".format(dem_diff_with_coreg),
                ".. image:: /{}".format(dem_diff_cdf_with_coreg),
                "",
            ]
        )
        src = "\n".join(
            [
                src,
                "**Generated coregistered DEMs:**",
                "",
                "* Tested Coreg DEM (COREG_DEM): {}".format(coreg_dsm_name),
                "   * dir: {} ".format(coreg_dsm_name_dir),
                "* Reference Coreg DEM (COREG_REF): {}".format(coreg_ref_name),
                "   * dir: {}".format(coreg_ref_name_dir),
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
                ".. image:: /{}".format(final_dem_diff_pdf),
                "",
            ]
        )

    # -> stats results table of contents
    src = "\n".join([src, "Stats Results", "===============", ""])
    if dem_diff_with_coreg:
        src = "\n".join(
            [
                src,
                "*Important: stats are generated on "
                + "COREG_REF - COREG_DEM difference*",
                "",
            ]
        )
    else:
        src = "\n".join(
            [
                src,
                "*Important: stats are generated on REF - DEM difference*",
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
    for partition_name, _ in partitions.items():
        src = "\n".join(
            [
                src,
                "* The :ref:`{partition_name} <{partition_name}>` ".format(
                    partition_name=partition_name
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
    for mode in modes:
        the_mode_pitch = modes_information[partition_name][mode]["pitch"]
        src = "\n".join(
            [
                src,
                "* The :ref:`{mode} <{mode}>` mode".format(mode=mode),
                "",
                "{} \n".format(the_mode_pitch),
                "",
            ]
        )

    # -> the results
    for partition_name, stats_results_d in partitions.items():
        src = "\n".join(
            [
                src,
                ".. _{}:".format(partition_name),
                "",
                "Classification layer: {}".format(partition_name),
                "{}-----------------------".format("-" * len(partition_name)),
                "",
            ]
        )
        # loop for results for each mode
        for mode in modes:
            if mode in stats_results_d:
                the_mode_histo = modes_information[partition_name][mode][
                    "histo"
                ]
                the_mode_fitted_histo = modes_information[partition_name][mode][
                    "fitted_histo"
                ]
                the_mode_csv = modes_information[partition_name][mode]["csv"]
            else:
                continue
            src = "\n".join(
                [
                    src,
                    "",
                    ".. _{}:".format(mode),
                    "",
                    "Mode: {}".format(mode),
                    "^^^^^^^^^^^^^^^{}".format("^" * len(mode)),
                    "",
                ]
            )
            # Histogram
            if the_mode_histo:
                src = "\n".join(
                    [
                        src,
                        "Graph showing mean and standard deviation",
                        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
                        ".. image:: /{}".format(the_mode_histo),
                        "",
                    ]
                )
            # Gaussian Fitted Histogram
            if the_mode_fitted_histo:
                src = "\n".join(
                    [
                        src,
                        "Fitted graph showing mean and standard deviation",
                        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
                        ".. image:: /{}".format(the_mode_fitted_histo),
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
                        "{}".format(the_mode_csv),
                        "",
                    ]
                )
    # Add source to the project
    spm.write_body(src)

    # Build the project
    try:
        spm.build_project("html")
    except Exception:
        print(
            ("Error when building report as {} output (ignored)".format("html"))
        )
        raise
    try:
        spm.build_project("latexpdf")
    except Exception:
        print(
            ("Error when building report as {} output (ignored)".format("pdf"))
        )

    # Sphinx project install
    spm.install_project()
