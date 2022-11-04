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
Mainly contains the SegmentationClassification class.
"""
import collections
import logging
import sys
from typing import Any, Dict

import xarray as xr

from .classification_layer import ClassificationLayer
from .classification_layer_template import ClassificationLayerTemplate


@ClassificationLayer.register("segmentation")
class SegmentationClassificationLayer(ClassificationLayerTemplate):
    """
    SegmentationClassificationLayer
    """

    def __init__(
        self,
        name: str,
        classification_layer_kind: str,
        cfg: Dict,
        dem: xr.Dataset = None,
    ):
        """
        Init function

        :param name: classification layer name
        :type name: str
        :param classification_layer_kind: classification layer kind
        :type classification_layer_kind: str
        :param cfg: layer's configuration
        :type cfg: Dict[str, Any]
        :param dem: dem
        :type dem:    xr.DataSet containing :

            - image : 2D (row, col) xr.DataArray float32
            - georef_transform: 1D (trans_len) xr.DataArray
            - classification_layer_masks : 3D (row, col, indicator) xr.DataArray
        :return: None
        """
        # Call generic init before supercharging
        super().__init__(name, classification_layer_kind, cfg, dem)

        # Classes
        self.classes: collections.OrderedDict = self.cfg["classes"]
        # Checking configuration during initialisation step
        # doesn't require classification layers
        if dem is not None:
            # Create labelled map to classification_layer from
            self._create_labelled_map()

            # Create class masks
            self._create_class_masks()

        logging.debug("ClassificationLayer created as: %s", self)

    def fill_conf_and_schema(
        self, cfg: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Add default values to the dictionary if there are missing
        elements and define the configuration schema

        :param cfg: coregistration configuration
        :type cfg: Dict[str, Any]
        :return cfg: coregistration configuration updated
        :rtype: Dict[str, Any]
        """
        cfg["classes"] = collections.OrderedDict(cfg["classes"])

        # Call generic fill_conf_and_schema
        cfg = super().fill_conf_and_schema(cfg)

        # Add subclass parameter to the default schema
        self.schema["classes"] = collections.OrderedDict
        self.check_classes(cfg)
        return cfg

    @staticmethod
    def check_classes(cfg: dict) -> None:
        """
        Verify users configuration for classes in segmentation classification
        :param cfg: segmentation configuration
        :type cfg: dict
        :return: None
        """
        classes_dict = cfg["classes"]
        if not all(
            isinstance(values, list) for values in classes_dict.values()
        ):
            logging.error(
                "Number associated to class in segmentation must be in a list"
            )
            sys.exit(1)
        if not all(
            isinstance(values[0], int) for values in classes_dict.values()
        ):
            logging.error(
                "Number associated to class in segmentation must be an int"
            )
            sys.exit(1)

    def _create_labelled_map(self):
        """
        Create the labelled map and save it if necessary
        :return: None
        """
        indicators = list(self.dem.classification_layer_masks.indicator.data)
        for idx, map_indicator in enumerate(indicators):
            if self.name in map_indicator:
                map_img = self.dem.classification_layer_masks.data[:, :, idx]
                # If support is included in the map_indicator, it will be placed
                # at its beggining as ref_ or sec_
                if "support_list" in self.dem.attrs:
                    support = self.dem.attrs["support_list"][idx]
                else:
                    support = "ref"
                # Store map_image
                self.map_image[support] = map_img
                # If _output_dir is set, create map_dataset and save
                if self._output_dir:
                    self.save_map_img(map_img, support)
