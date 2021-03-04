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
Create sphinx report and compile it for html and pdf format
"""

# Standard imports
import collections
import csv
import fnmatch
import glob
import os
import sys

# DEMcompare imports
from .sphinx_project_generator import SphinxProjectManager


def recursive_search(directory, pattern):
    """
    Recursively look up pattern filename into dir tree

    :param directory:
    :param pattern:
    :return:
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


def generate_report(  # noqa: C901
    working_dir,
    dsm_name,
    ref_name,
    partitions=None,
    doc_dir=".",
    project_dir=".",
):
    """
    Create pdf report from png graph and csv stats summary

    :param working_dir: directory in which to find *mode*.png
                    and *mode*.csv files for each mode in modename
    :param dsm_name: name or path to the dsm to be compared against the ref
    :param ref_name: name or path to the reference dsm
    :param partitions: list of partition, contains modes by partition
    :param doc_dir: directory in which to find the output documentation
    :param project_dir:
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

    for partition_name, _ in partitions.items():
        # Initialize mode informations for partition
        modes_information[partition_name] = collections.OrderedDict()
        modes_information[partition_name]["standard"] = {
            "pitch": "This mode results simply relies only on valid values. "
            "This means nan values "
            "(whether they are from the error image or the reference support "
            "image when do_classification is on), but also ouliers and "
            "masked ones has been discarded."
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
                    with open(result[0], "r") as csv_file:
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

    # Find DSMs differences
    dem_diff_without_coreg = recursive_search(
        working_dir, "initial_dem_diff.png"
    )[0]
    result = recursive_search(working_dir, "final_dem_diff.png")
    if len(result) > 0:
        dem_diff_with_coreg = result[0]
    else:
        dem_diff_with_coreg = None

    # Create source
    # -> header part
    src = "\n".join(
        [
            ".. _DSM_COMPARE_REPORT:",
            "",
            "*********************",
            " DSM COMPARE REPORT",
            "*********************" "",
            "**It shows comparison results between the following DSMs:**",
            "",
            "* **The DSM to evaluate**: {}".format(dsm_name),
            "* **The Reference DSM**  : {}".format(ref_name),
            "",
        ]
    )
    # -> DSM differences
    src = "\n".join(
        [
            src,
            "**Below is shown elevation differences between both DSMs:**",
            "",
            # 'DSM diff without coregistration',
            # '-------------------------------',
            ".. image:: /{}".format(dem_diff_without_coreg),
            "",
        ]
    )
    if dem_diff_with_coreg:
        src = "\n".join(
            [
                src,
                # 'DSM diff with coregistration',
                # '----------------------------',
                ".. image:: /{}".format(dem_diff_with_coreg),
                "",
            ]
        )
    # -> table of contents
    src = "\n".join(
        [
            src,
            "**The comparison outcomes are provided for the evaluation \
                mode listed hereafter:**",
            "",
        ]
    )
    for mode in modes:
        src = "\n".join(
            [src, "* The :ref:`{mode} <{mode}>` mode".format(mode=mode)]
        )

    # -> the results
    for partition_name, stats_results_d in partitions.items():
        src = "\n".join(
            [
                src,
                "*{} classification layer*".format(partition_name),
                "{}".format("#" * len(partition_name)),
            ]
        )

        for mode in modes:
            if mode in stats_results_d:
                the_mode_pitch = modes_information[partition_name][mode][
                    "pitch"
                ]
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
                    "*Results for the {} evaluation mode*".format(mode),
                    "=================================={}".format(
                        "=" * len(mode)
                    ),
                    "",
                    "{}".format(the_mode_pitch),
                    "",
                ]
            )
            if the_mode_histo:
                src = "\n".join(
                    [
                        src,
                        # 'Graph showing mean and standard deviation',
                        # '-----------------------------------------',
                        ".. image:: /{}".format(the_mode_histo),
                        "",
                    ]
                )
            if the_mode_fitted_histo:
                src = "\n".join(
                    [
                        src,
                        # 'Fitted graph showing mean and standard deviation',
                        # '-----------------------------------------',
                        ".. image:: /{}".format(the_mode_fitted_histo),
                        "",
                    ]
                )
            if the_mode_csv:
                src = "\n".join(
                    [
                        src,
                        # 'Table showing comparison metrics',
                        # '--------------------------------',
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
