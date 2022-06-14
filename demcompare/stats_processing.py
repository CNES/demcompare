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
Mainly contains the StatsProcessing class
for stats computation of an input dem
"""
# pylint:disable=no-member

import copy

# Standard imports
import logging
import sys
import traceback
from typing import Dict, List, Union

# Third party imports
import xarray as xr

from demcompare.classification_layer import (
    ClassificationLayer,
    FusionClassificationLayer,
)
from demcompare.metric import Metric

from .initialization import ConfigType
from .stats_dataset import StatsDataset


class StatsProcessing:
    """
    StatsProcessing class
    """

    # Default parameters in case they are not specified in the cfg
    # Save results option
    _SAVE_RESULTS = False
    # Default segmentation layer
    _DEFAULT_GLOBAL_LAYER_NAME = "global"
    _DEFAULT_GLOBAL_LAYER = {
        "type": "global",
    }
    # Plot real histograms option
    _PLOT_REAL_HISTS = False
    # Remove outliers option
    _REMOVE_OUTLIERS = False

    # Initialization
    def __init__(self, cfg: Dict, dem: xr.Dataset):
        """
        Initialization of a StatsProcessing object

        :param cfg: cfg
        :type cfg: Dict
        :param dem: dem
        :type dem:    xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layers : 3D (row, col, nb_classif)
                  xr.DataArray float32
        :return: None
        """
        # Cfg
        cfg = self.fill_conf(cfg)
        self.cfg = cfg
        # Output directory
        self.output_dir = self.cfg["output_dir"]
        # Remove outliers option
        self.remove_outliers = self.cfg["remove_outliers"]
        # Save results boolean
        self.save_results = self.cfg["save_results"]
        # Input dem
        self.dem = dem

        # Initialize StatsDataset object
        self.stats_dataset = StatsDataset(self.dem["image"].data)
        # Classification layers
        self.classification_layers = []
        # Classification layers names
        self.classification_layers_names = []
        # Create classification layers
        self._create_classif_layers()

    def fill_conf(
        self, cfg: ConfigType = None
    ):  # pylint:disable=too-many-branches
        """
        Init Stats options from configuration

        :param cfg: Input demcompare configuration
        :type cfg: ConfigType
        """
        # Initialize if cfg is not defined
        if cfg is None:
            cfg = {}
        if "classification_layers" not in cfg:
            cfg["classification_layers"] = {}
        # Add default global layer
        cfg["classification_layers"][
            self._DEFAULT_GLOBAL_LAYER_NAME
        ] = copy.deepcopy(self._DEFAULT_GLOBAL_LAYER)
        # If metrics have been specified,
        # add them to all classif layers
        if "metrics" in cfg:
            for _, classif_cfg in cfg["classification_layers"].items():
                if "metrics" in classif_cfg:
                    for new_metric in cfg["metrics"]:
                        classif_cfg["metrics"].append(new_metric)
                else:
                    classif_cfg["metrics"] = cfg["metrics"]

        # Give the default value if the required element
        # is not in the configuration
        if "remove_outliers" in cfg:
            cfg["remove_outliers"] = cfg["remove_outliers"] == "True"
        else:
            cfg["remove_outliers"] = self._REMOVE_OUTLIERS
        if "save_results" in cfg:
            cfg["save_results"] = cfg["save_results"] == "True"
        else:
            cfg["save_results"] = self._SAVE_RESULTS

        if "output_dir" not in cfg:
            cfg["output_dir"] = None
            if cfg["save_results"]:
                logging.error(
                    "save_results option is activated"
                    " but no output_dir has been set. "
                    "Please set the output_dir parameter or deactivate"
                    " the saving options."
                )
                sys.exit(1)

        return cfg

    def _create_classif_layers(self):
        """
        Create the classification layer objects
        """

        # Loop over cfg's classification_layers
        if "classification_layers" in self.cfg:
            for name, clayer in self.cfg["classification_layers"].items():
                # Fusion layer must be created once all
                # classifications are created
                if name == "fusion":
                    continue
                try:
                    # Set output_dir and save_results on the classif
                    # layer's cfg
                    clayer["output_dir"] = self.output_dir
                    clayer["save_results"] = str(self.save_results)
                    # If outliers handling has not been specified
                    # on the classification layer cfg,
                    # add the global statistics one
                    if "remove_outliers" not in clayer:
                        clayer["remove_outliers"] = str(self.remove_outliers)
                    # Create ClassificationLayer object
                    self.classification_layers.append(
                        ClassificationLayer(
                            name,
                            clayer["type"],
                            self.dem,
                            clayer,
                        )
                    )
                    self.classification_layers_names.append(name)
                except ValueError as error:
                    traceback.print_exc()
                    logging.error(
                        (
                            "Cannot create classification_layer"
                            " for {}:{} -> {}".format(
                                clayer["name"], clayer, error
                            )
                        )
                    )
        # Compute fusion layer it specified in the conf
        if "fusion" in list(self.cfg["classification_layers"].keys()):
            for support, classif_names in self.cfg["classification_layers"][
                "fusion"
            ].items():
                layers_to_fusion = []
                for name in classif_names:
                    layers_to_fusion.append(
                        self.classification_layers[
                            self.classification_layers_names.index(name)
                        ]
                    )
                if support == "ref":
                    support_idx = 0
                else:
                    support_idx = 1
                self.classification_layers.append(
                    FusionClassificationLayer(layers_to_fusion, support_idx)
                )
                self.classification_layers_names.append(
                    self.classification_layers[-1].name
                )

        for classif in self.classification_layers:
            logging.debug("List of classification layers: {}".format(classif))

    def compute_stats(
        self,
        classification_layer: List[str] = None,
        metrics: List[Union[dict, str]] = None,
    ) -> StatsDataset:
        """
        Compute DEM stats

        If no classification layer is given, all classification
        layers are considered
        If no metrics are given, all metrics are considered

        :param classification_layer: names of the layers
        :type classification_layer: List[str]
        :param metrics: List of metrics to be computed
        :type metrics: List[Union[dict, str]]
        :return: stats dataset
        :rtype: StatsDataset
        """
        # Select input classification layers
        selected_classif_layers = []
        if classification_layer:
            for name in classification_layer:
                idx = self.classification_layers_names.index(name)
                selected_classif_layers.append(self.classification_layers[idx])
        else:
            # If no classification layer is specified, select all layers
            selected_classif_layers = self.classification_layers

        # For each selected classification layer
        for classif in selected_classif_layers:
            # Compute and fill the corresponding
            # stats_dataset classification layer's xr.Dataset stats
            logging.info(
                "Computing classification layer {} stats...".format(
                    classif.name
                )
            )

            classif.compute_classif_stats(
                self.dem,
                self.stats_dataset,
                metrics=metrics,
            )

        return self.stats_dataset

    @staticmethod
    def show_available_metrics():
        print(Metric.available_metrics)

    def show_available_classification_layers(self):
        print(self.classification_layers_names)
