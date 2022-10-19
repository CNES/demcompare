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
from typing import Dict

import xarray as xr

from ..helpers_init import ConfigType
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
        dem: xr.Dataset,
        cfg: Dict,
    ):
        """
        Init function

        :param name: classification layer name
        :type name: str
        :param classification_layer_kind: classification layer kind
        :type classification_layer_kind: str
        :param dem: dem
        :type dem:    xr.DataSet containing :

            - image : 2D (row, col) xr.DataArray float32
            - georef_transform: 1D (trans_len) xr.DataArray
            - classification_layer_masks : 3D (row, col, indicator) xr.DataArray
        :param cfg: layer's configuration
        :type cfg: ConfigType
        :return: None
        """
        # Call generic init before supercharging
        super().__init__(name, classification_layer_kind, dem, cfg)

        # Classes
        self.classes: collections.OrderedDict = self.cfg["classes"]
        # Create labelled map to classification_layer from
        self._create_labelled_map()

        # Create class masks
        self._create_class_masks()

        logging.debug("ClassificationLayer created as: %s", self)

    def fill_conf_and_schema(self, cfg: ConfigType = None) -> ConfigType:
        """
        Add default values to the dictionary if there are missing
        elements and define the configuration schema

        :param cfg: coregistration configuration
        :type cfg: ConfigType
        :return cfg: coregistration configuration updated
        :rtype: ConfigType
        """
        cfg["classes"] = collections.OrderedDict(cfg["classes"])

        # Call generic fill_conf_and_schema
        cfg = super().fill_conf_and_schema(cfg)

        # Add subclass parameter to the default schema
        self.schema["classes"] = collections.OrderedDict
        return cfg

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
