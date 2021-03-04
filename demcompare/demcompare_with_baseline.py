#!/usr/bin/env python
# coding: utf8
# PYTHON_ARGCOMPLETE_OK
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
Tests : Compare results against baseline
"""

# Standard imports
import argparse
import glob
import json
import os
from collections import OrderedDict

# Third party imports
import argcomplete

# DEMcompare imports
from demcompare.output_tree_design import get_out_dir


def load_json(json_file):
    with open(json_file, "r") as file:
        return json.load(file)


def load_csv(csv_file):
    with open(csv_file, "r") as file:
        return file.readlines()


def check_csv(csv_ref, csv_test, csv_file, epsilon):
    """
    Check CSV function
    """
    if csv_ref[0] != csv_test[0]:
        raise ValueError(
            "Inconsistent stats between baseline ({}) "
            "and tested version ({}) for file {}".format(
                csv_ref[0], csv_test[0], csv_file
            )
        )

    csv_differences = []
    # - first row of csv file is titles (hence csv[1:len(csv)])
    for row_ref, row_test in zip(
        csv_ref[1 : len(csv_ref)], csv_test[1 : len(csv_test)]
    ):
        # - we need to split a row by ',' to get columns
        # after we removed the '\r\n' end characters
        cols_ref = row_ref.strip("\r\n").split(",")
        cols_test = row_test.strip("\r\n").split(",")

        # - test if class are the same (first column is class name)
        if cols_ref[0] != cols_test[0]:
            raise ValueError(
                "Inconsistent class name for file {} between baseline ({}) "
                "and tested version ({})".format(
                    csv_file, cols_ref[0], cols_test[0]
                )
            )

        # - first column is class name, and then we have to cast values in float
        f_cols_ref = [
            float(col_value) for col_value in cols_ref[1 : len(cols_ref)]
        ]
        f_cols_test = [
            float(col_value) for col_value in cols_test[1 : len(cols_test)]
        ]

        # see if we differ by more than epsilon
        results = [
            abs(ref - test) <= epsilon
            for ref, test in zip(f_cols_ref, f_cols_test)
        ]
        if sum(results) != len(results):
            # then we have some false values
            indices = [i for i, item in enumerate(results) if item is False]
            for index in indices:
                diff = OrderedDict()
                diff["csv_file"] = csv_file
                diff["class name"] = cols_ref[0]
                diff["stat name"] = (
                    csv_ref[0].strip("\r\n").split(",")[index + 1]
                )
                diff["baseline_val"] = f_cols_ref[index]
                diff["test_val"] = f_cols_test[index]
                csv_differences.append(diff)
    return csv_differences


def run(baseline_dir, output_dir, epsilon=1.0e-6):
    """
    Compare output_dir results to baseline_dir ones

    :param baseline_dir:
    :param output_dir:
    :param epsilon:
    :return:
    """

    # read both json files
    baseline_fjson = load_json(os.path.join(baseline_dir, "final_config.json"))
    output_fjson = load_json(os.path.join(output_dir, "final_config.json"))

    # get both stats dir
    baseline_statsdir = os.path.join(
        baseline_dir,
        get_out_dir(
            "stats_dir",
            design=(
                baseline_fjson["otd"] if "otd" in baseline_fjson else "raw_OTD"
            ),
        ),
    )
    output_statsdir = os.path.join(
        output_dir,
        get_out_dir("stats_dir", design=output_fjson["otd"]),
        "slope",
    )

    # check csv files consistency
    ext = ".csv"
    baseline_csv_files = glob.glob("{}/*{}".format(baseline_statsdir, ext))
    output_csv_files = glob.glob("{}/*{}".format(output_statsdir, ext))
    baseline_data = [load_csv(csv_file) for csv_file in baseline_csv_files]
    test_data = [load_csv(csv_file) for csv_file in output_csv_files]

    # before checking values we see if class names (slope range)
    # and stats tested are the same between both versions
    if len(baseline_data) != len(test_data):
        raise ValueError(
            "Inconsistent number of csv files between baseline ({}) "
            "and tested output ({})".format(len(baseline_data), len(test_data))
        )

    # for each csv file
    differences = [
        check_csv(csv_ref, csv_test, csv_file, epsilon)
        for csv_ref, csv_test, csv_file in zip(
            baseline_data, test_data, baseline_csv_files
        )
    ]

    if sum(len(diff) for diff in differences) != 0:
        error = "Invalid results obtained with this version \
                 of demcompare: \n{}".format(
            differences
        )
        raise ValueError(error)

    print(
        (
            "No difference between tested files : {}".format(
                list(zip(baseline_csv_files, output_csv_files))
            )
        )
    )


def get_parser():
    """
    ArgumentParser for compare_with_baseline
    :param None
    :return parser
    """
    parser = argparse.ArgumentParser(
        description=("Compares demcompare test_config.json outputs to baseline")
    )

    parser.add_argument(
        "--baselinePath", default="./test_baseline", help="path to the baseline"
    )
    parser.add_argument(
        "--currentRunPath",
        default="./test_output",
        help="path to the demcompare run to test against the baseline",
    )

    return parser


def main():
    """
    Call demcompare_with_baseline's main
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    run(args.baselinePath, args.currentRunPath)


if __name__ == "__main__":
    main()
