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
from typing import Any, Dict, List

# Third party imports
import numpy as np
from json_checker import Or

from .classification_layer import ClassificationLayer
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
        classification_layers: List[ClassificationLayer],
        support: str,
        name: str,
        metrics: List = None,
    ):
        """
        :param classification_layers: list of ClassificationLayers
        :type classification_layers: List[ClassificationLayerTemplate]
        :support: support dem, ref or sec
        :type support: str
        :name: layer name
        :type name: str
        :metrics: optional input metrics
        :type metrics: List
        """

        # If only one classification_layer is given, raise error
        if len(classification_layers) == 1:
            logging.error(
                "There must be at least 2"
                " classification_layer_masks"
                " to be merged together"
            )
            raise self.NotEnoughDataToClassificationLayerError
        # Store classification layers
        self.classification_layers = classification_layers
        # Initialize fusion conf
        cfg = self.fill_fusion_conf(metrics)
        # Support
        self.support = support
        # Name
        self.name = name
        super().__init__(
            name=self.name,
            classification_layer_kind="classification_layer_masks",
            dem=classification_layers[0].dem,
            cfg=cfg,
        )
        # Checking configuration during initialisation step
        # doesn't require classification layers
        if classification_layers[0].dem is not None:
            # Create classification_layer classes
            # (labelled map with associated classes)
            self._merge_classes_and_create_classes_masks()
            self._create_labelled_map()

        logging.debug("ClassificationLayer created as: %s", self)

    def fill_fusion_conf(self, metrics: List = None) -> Dict[str, Any]:
        """
        Fill the fusion layer configuration

        :param metrics: optinal input metrics
        :type metrics: List
        :return cfg: configuration updated
        :rtype: Dict[str, Any]
        """
        # Initialize cfg layer with necessary parameters
        cfg: Dict = {}
        cfg["remove_outliers"] = str(
            self.classification_layers[0].remove_outliers
        )
        cfg["output_dir"] = self.classification_layers[0]._output_dir
        cfg["type"] = "fusion"
        cfg["nodata"] = self.classification_layers[0].nodata
        # If metrics have been defined, add them
        cfg["metrics"] = metrics

        self.schema = {
            "output_dir": Or(str, None),
            "nodata": Or(float, int),
            "type": "fusion",
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
            map_fusion[
                np.where(self.classes_masks[self.support][idx])
            ] = class_item
        # Add map_fusion on the map_image
        self.map_image[self.support] = map_fusion
        # Save results
        if self._output_dir:
            self.save_map_img(map_fusion, self.support)

    def _merge_classes_and_create_classes_masks(self):
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
                    class_mask = dict_classification_layers[
                        layer_name
                    ].classes_masks[self.support][label_idx]

                    # Resulting mask is the superposition of
                    # all combined layer's mask
                    masks = masks * class_mask

                # Append new classe's support mask
                support_masks.append(masks)
            self.classes_masks[self.support] = support_masks

    @staticmethod
    def _create_merged_classes(
        classification_layers: List[ClassificationLayer],
    ):
        """
        Generate the 'classes' dictionary for merged layers
        :param classification_layers: list of classes to merge
        :type classification_layers: List[ClassificationLayer]
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
