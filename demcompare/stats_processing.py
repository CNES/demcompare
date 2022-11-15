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

import copy

# Standard imports
import logging
import os
import traceback
from typing import Any, Dict, List, Union

# Third party imports
import xarray as xr

from demcompare.classification_layer import (
    ClassificationLayer,
    FusionClassificationLayer,
)
from demcompare.metric import Metric
from demcompare.output_tree_design import get_out_dir

from .stats_dataset import StatsDataset


class StatsProcessing:
    """
    StatsProcessing class
    """

    # Default parameters in case they are not specified in the cfg
    # Default global layer
    _DEFAULT_GLOBAL_LAYER_NAME = "global"
    _DEFAULT_GLOBAL_LAYER = {
        "type": "global",
    }
    # Remove outliers option
    _REMOVE_OUTLIERS = False
    # Default metrics for input alti_diff
    _DEFAULT_METRICS_ALTI_DIFF = {
        "metrics": [
            "mean",
            "median",
            "max",
            "min",
            "sum",
            {"percentil_90": {"remove_outliers": "False"}},
            "squared_sum",
            "nmad",
            "rmse",
            "std",
        ]
    }
    # Default metrics for a single input dem
    _DEFAULT_METRICS = {
        "metrics": [
            "mean",
            "median",
            "max",
            "min",
            "sum",
            "squared_sum",
            "std",
        ]
    }
    # Initialization

    def __init__(
        self, cfg: Dict, dem: xr.Dataset = None, input_diff: bool = False
    ):
        """
        Initialization of a StatsProcessing object

        :param cfg: cfg
        :type cfg: Dict
        :param dem: dem
        :type dem:    xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, nb_classif)
                  xr.DataArray float32
        :param input_diff: if the input dem is an altitude difference
        :type input_diff: bool
        :return: None
        """
        # Cfg
        cfg = self.fill_conf(cfg, input_diff)
        self.cfg: Dict = cfg
        # Output directory
        self.output_dir: Union[str, None] = self.cfg["output_dir"]
        if self.output_dir:
            # Create plots dir
            self._plots_dir = os.path.join(
                self.output_dir, get_out_dir("snapshots_dir")
            )
            os.makedirs(self._plots_dir, exist_ok=True)

        # Remove outliers option
        self.remove_outliers: bool = self.cfg["remove_outliers"]
        # Input dem
        self.dem: xr.Dataset = dem
        # Classification layers
        self.classification_layers: List[ClassificationLayer] = []
        # Classification layers names
        self.classification_layers_names: List[str] = []
        # Initialise and test parameters for StatProcessing object
        if dem is None:
            # Create classification layers
            self._create_classif_layers()
        else:
            # Initialize StatsDataset object
            self.stats_dataset: StatsDataset = StatsDataset(
                self.dem["image"].data
            )
            # Create classification layers
            self._create_classif_layers()

    def fill_conf(
        self, cfg: Dict[str, Any] = None, input_diff: bool = False
    ):  # pylint:disable=too-many-branches
        """
        Init Stats options from configuration

        :param cfg: Input demcompare configuration
        :type cfg: Dict[str, Any]
        :param input_diff: If the input parameter is an altitude difference
        :type input_diff: bool
        """

        # Initialize if cfg is not defined
        if cfg is None:
            cfg = {}
        # Initialize if classification_layers is not defined
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
        # If no metrics have been specified on any level
        # for a classification layer, add the default metrics
        # according to the input dem type
        else:
            for _, classif_cfg in cfg["classification_layers"].items():
                if "metrics" not in classif_cfg:
                    if input_diff:
                        classif_cfg.update(self._DEFAULT_METRICS_ALTI_DIFF)
                    else:
                        classif_cfg.update(self._DEFAULT_METRICS)

        # Give the default value if the required element
        # is not in the configuration
        if "remove_outliers" in cfg:
            cfg["remove_outliers"] = cfg["remove_outliers"] == "True"
        else:
            cfg["remove_outliers"] = self._REMOVE_OUTLIERS
        if "output_dir" not in cfg:
            cfg["output_dir"] = None
        return cfg

    def _create_classif_layers(self):
        """
        Create the classification layer objects
        """

        # Loop over cfg's classification_layers
        fusion_layers = []
        if "classification_layers" in self.cfg:
            for name, clayer in self.cfg["classification_layers"].items():
                # Fusion layer must be created once all
                # classifications are created
                if clayer["type"] == "fusion":
                    fusion_layers.append(name)
                    continue
                try:
                    # Set output_dir on the classif
                    # layer's cfg
                    clayer["output_dir"] = self.output_dir
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
                            clayer,
                            self.dem,
                        )
                    )
                    self.classification_layers_names.append(name)
                except ValueError as error:
                    traceback.print_exc()
                    logging.error(
                        (
                            "Cannot create classification_layer"
                            " for %s:%s -> %s",
                            clayer["name"],
                            clayer,
                            error,
                        )
                    )
        # Compute fusion layer it specified in the conf
        # Fusion layers specify its support on the input cfg
        for fusion_name in fusion_layers:
            # Copy to suppress the metrics information
            tmp_fusion_cfg = copy.deepcopy(
                self.cfg["classification_layers"][fusion_name]
            )
            # Supress type parameter from the dict
            tmp_fusion_cfg.pop("type")
            # Get fusion metrics if present in the conf
            fusion_metrics = None
            if "metrics" in tmp_fusion_cfg:
                fusion_metrics = tmp_fusion_cfg.pop("metrics")
            for support, classif_names in tmp_fusion_cfg.items():
                # Add the layers to be fusionned from the conf
                layers_to_fusion = []
                for name in classif_names:
                    layers_to_fusion.append(
                        self.classification_layers[
                            self.classification_layers_names.index(name)
                        ]
                    )
                # Create fusion layer
                self.classification_layers.append(
                    FusionClassificationLayer(  # type:ignore
                        layers_to_fusion, support, fusion_name, fusion_metrics
                    )
                )
                # Add fusion layer name on the classif_layers_names
                self.classification_layers_names.append(
                    self.classification_layers[-1].name
                )

        for classif in self.classification_layers:
            logging.debug("List of classification layers: %s", classif)

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
            logging.debug(
                "Computing classification layer %s stats...", classif.name
            )

            classif.compute_classif_stats(
                self.dem, self.stats_dataset, metrics=metrics
            )

        return self.stats_dataset

    @staticmethod
    def show_all_available_metrics() -> str:
        """
        Return a string showing all available values
        :return: output_metrics
        :rtype: str
        """
        available_metrics = list(Metric.available_metrics.keys())
        output_metrics = f"{available_metrics}"
        return output_metrics

    def show_available_classification_layers(self) -> list:
        return self.classification_layers_names
