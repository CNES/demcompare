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
"""
Logconf demcompare:
contains logging configuration
"""

import json

# Standard imports
import logging
import os
from datetime import datetime


def setup_logging(
    logconf_path="logging.json",
    default_level=logging.INFO,
):
    """
    Setup the logging configuration
    If logconf_path is found, set the json logging configuration
    Else put default_level

    :param logconf_path: path to the configuration file
    :type logconf_path: string
    :param default_level: default level
    :type default_level: logging level
    :param cfg: input cfg
    :type cfg: Dict
    """
    logconf_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), logconf_path)
    )
    if os.path.exists(logconf_path):
        with open(logconf_path, "rt", encoding="utf8") as logconf_file:
            config = json.load(logconf_file)
        # Set config and force default_level from command_line
        logging.config.dictConfig(config)
        logging.getLogger().setLevel(default_level)
    else:
        # take default python config with default_level from command line
        logging.basicConfig(level=default_level)


def add_log_file(out_dir: str):
    """
    Add dated file handler to the logger.

    :param out_dir: output directory in which the log file will be created
    :type out_dir: str
    :returns: None
    """
    # set file log handler
    now = datetime.now()
    date = now.strftime("%y-%m-%d_%Hh%Mm")
    h_log_file = logging.FileHandler(os.path.join(out_dir, f"{date}_logs.log"))

    # add it to the logger
    logging.getLogger().addHandler(h_log_file)
