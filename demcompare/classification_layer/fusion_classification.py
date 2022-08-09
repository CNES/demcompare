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
Mainly contains the FussionClassification class.
"""
import collections
import itertools
import logging
from typing import List

# Third party imports
import numpy as np
from json_checker import Or

from ..initialization import ConfigType
from .classification_layer_template import ClassificationLayerTemplate


class FusionClassificationLayer(ClassificationLayerTemplate):
    """
    FusionClassificationLayer is the fusion of more than one
    ClassificationLayer, in order to retrieve the information
    of pixels belonging to the intersections of two classes
    of different ClassificationLayers.

    """

    # pylint: disable=protected-access

    def __init__(
        self,
        classification_layers: List[ClassificationLayerTemplate],
        map_idx: int,
    ):
        """
        :param classification_layers: list of ClassificationLayers
        :type classification_layers: List[ClassificationLayerTemplate]
        :map_idx: index map to fusion from each ClassificationLayer
        :type map_idx: int
        """

        # If only one classification_layer is given, raise error
        if len(classification_layers) == 1:
            logging.error(
                "There must be at least 2"
                " classification_layers"
                " to be merged together"
            )
            raise self.NotEnoughDataToClassificationLayerError
        # Store classification layers
        self.classification_layers = classification_layers
        # Initialize and fill cfg
        cfg = self.fill_conf_and_schema()
        # Fusion layer name
        self.name = "fusion_layer" + str(map_idx)

        super().__init__(
            name=self.name,
            classification_layer_kind="classification_layers",
            dem=classification_layers[0].dem,
            cfg=cfg,
        )

        # Create classification_layer classes
        # (labelled map with associated classes)
        self._merge_classes_and_create_classes_masks(map_idx)
        self._create_labelled_map()

        logging.info("ClassificationLayer created as: {}".format(self))

    def fill_conf_and_schema(self, cfg: ConfigType = None) -> ConfigType:
        """
        Add default values to the dictionary if there are missing
        elements and define the configuration schema

        :param cfg: coregistration configuration
        :type cfg: ConfigType
        :return cfg: coregistration configuration updated
        :rtype: ConfigType
        """
        # Initialize cfg layer with necessary parameters
        cfg = {}
        cfg["save_results"] = self.classification_layers[0].save_results
        cfg["remove_outliers"] = self.classification_layers[0].remove_outliers
        cfg["output_dir"] = self.classification_layers[0]._output_dir
        cfg["type"] = "segmentation"
        cfg["no_data"] = self.classification_layers[0].nodata
        if "metric" not in cfg:
            cfg.update(self._DEFAULT_METRICS)

        self.schema = {
            "save_results": bool,
            "output_dir": Or(str, None),
            "no_data": Or(float, int),
            "type": "segmentation",
            "metrics": list,
            "remove_outliers": bool,
        }

        return cfg

    def _create_labelled_map(self):
        """
        Create the labelled map
        :return: None
        """
        # Get dems shape
        dems_shape = self.dem["image"].data.shape

        # Initialize fusion map
        map_fusion = np.ones(dems_shape) * self.nodata
        for idx, (_, class_item) in enumerate(self.classes.items()):
            # Fill fusion map with classes masks
            map_fusion[np.where(self.classes_masks[0][idx])] = class_item
        # Add map_fusion on the map_image
        self.map_image.append(map_fusion)
        # Save results
        if self.save_results:
            self.save_map_img(map_fusion, 0)

    def _merge_classes_and_create_classes_masks(self, map_idx):
        """
        Merge classes of the classification layers
        and create the classes_masks
        :return: None
        """
        # Create all combinations and new classes
        all_combi_labels, self.classes = self._create_merged_classes(
            self.classification_layers
        )
        # Dem shape
        dems_shape = self.dem["image"].data.shape
        # Create dict to easily access each classification layer
        dict_classification_layers = {
            classif.name: classif for classif in self.classification_layers
        }
        # Iterate over the fusionned maps
        for _ in self.classification_layers:
            # Initialize support masks
            support_masks = []
            # Iterate over all combined layers
            for combi in all_combi_labels:
                # Initialize new layer mask
                masks = np.ones(dems_shape) * True
                for elm in combi:
                    layer_name = elm[0]
                    label_idx = elm[1]
                    if map_idx == 0:
                        class_mask = dict_classification_layers[
                            layer_name
                        ].classes_masks[map_idx][label_idx]
                    # If map_idx is 1, access the maximum available
                    # class mask (1 if both are defined,
                    # 0 if only one is defined)
                    else:
                        class_mask = dict_classification_layers[
                            layer_name
                        ].classes_masks[-1][label_idx]

                    # Resulting mask is the superposition of
                    # all combined layer's mask
                    masks = masks * class_mask

                # Append new classe's support mask
                support_masks.append(masks)
            self.classes_masks.append(support_masks)

    @staticmethod
    def _create_merged_classes(
        classification_layers: List[ClassificationLayerTemplate],
    ):
        """
        Generate the 'classes' dictionary for merged layers
        :param classification_layers: list of classes to merge
        :type classification_layers: List[ClassificationLayerTemplate]
        :return:
        """
        # Initialize list of all classes to be combined
        classes_to_merge = []
        # Iterate over classification layers
        for classification_layer in classification_layers:
            # Add all the classes of each classification layer
            classes_to_merge.append(
                [
                    (
                        classification_layer.name,
                        idx,
                        class_name + ":" + str(class_item),
                    )
                    for idx, (class_name, class_item) in enumerate(
                        classification_layer.classes.items()
                    )
                ]
            )
        # Compute all possible combinations
        all_combi_labels = list(itertools.product(*classes_to_merge))
        # Initialize new classes
        new_classes = collections.OrderedDict()
        for idx, combi in enumerate(all_combi_labels):
            # Create new label inside new_classes dict
            new_label_name = "_&_".join(
                [
                    "_".join([name, str(label).split(":", maxsplit=1)[0]])
                    for name, value, label in combi
                ]
            )
            # To improve labelled map visualization,
            # new class values start at value 1, not at 0
            new_classes[new_label_name] = idx + 1

        return all_combi_labels, new_classes
