#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
demcompare aims at coregistering and comparing two dsms
"""

from __future__ import print_function
import argparse
import copy

import demcompare

DEFAULT_STEPS = ['coregistration', 'stats', 'report']
ALL_STEPS = copy.deepcopy(DEFAULT_STEPS)


def get_parser():
    """
    ArgumentParser for demcompare
    :param None
    :return parser
    """
    parser = argparse.ArgumentParser(description=('Compares DSMs'))

    parser.add_argument('config', metavar='config.json',
                        help=('path to a json file containing the paths to '
                              'input and output files and the algorithm '
                              'parameters'))
    parser.add_argument('--step', type=str, nargs='+', choices=ALL_STEPS,
                        default=DEFAULT_STEPS)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--display', action='store_true')

    return parser

def main():
    """
    Call demcompare's main
    """
    parser = get_parser()
    args = parser.parse_args()
    demcompare.run(args.config, args.step, debug=args.debug, display=args.display)

if __name__ == "__main__":
    main()
