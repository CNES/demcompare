#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
Compare results again baseline

"""

import json
import argparse
import glob
import os


def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def main():
    output_dir='../test_output/'
    baseline_dir='../test_baseline/'

    # Check json files consistency
    ext='.json'
    json_files = glob.glob('{}/*{}'.format(baseline_dir,ext))
    baseline_data = [load_json(json_file) for json_file in json_files]
    test_data = [load_json(os.path.join(baseline_dir,os.path.basename(json_file))) for json_file in json_files]
    results = [a==b for a,b in zip(baseline_data, test_data)]
    if sum(results) != len(results):
        raise ValueError('invalid results obtained with this version of dem_compare.py')

def get_parser():
    """
    ArgumentParser for compare_with_baseline
    :param None
    :return parser
    """
    parser = argparse.ArgumentParser(description=('Compares dem_compare.py test_config.json outputs to baseline'))

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main()
