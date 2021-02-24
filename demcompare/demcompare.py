#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# PYTHON_ARGCOMPLETE_OK

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
demcompare aims at coregistering and comparing two dsms
"""

from __future__ import print_function
import argcomplete, argparse
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
                        default=DEFAULT_STEPS,
                        help='steps to choose. default: all steps')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode')
    parser.add_argument('--display', action='store_true',
                        help='choose between plot show and plot save. '
                             'default: plot save')
    parser.add_argument('--version', '-v', action='version',
                        version='%(prog)s {version}'.format(
                                        version=demcompare.__version__))
    return parser

def main():
    """
    Call demcompare's main
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    demcompare.run(args.config, args.step, debug=args.debug, display=args.display)

if __name__ == "__main__":
    main()
