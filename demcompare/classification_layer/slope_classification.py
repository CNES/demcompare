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
Mainly contains the SlopeClassification class.
"""
import collections
import logging
import sys
from typing import Any, Dict, List

import numpy as np
import xarray as xr

# DEMcompare imports
from demcompare.dem_tools import create_dem

from .classification_layer import ClassificationLayer
from .classification_layer_template import ClassificationLayerTemplate

# Third party imports


@ClassificationLayer.register("slope")
class SlopeClassificationLayer(ClassificationLayerTemplate):
    """
    SlopeClassificationLayer
    """

    _RANGES = [0, 5, 10, 25, 45]

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
            - classification_layer_masks : 3D (row, col, indicator)
             xr.DataArray
        :return: None
        """
        # Call generic init before supercharging
        super().__init__(name, classification_layer_kind, cfg, dem)

        # Ranges
        self.ranges: List = self.cfg["ranges"]
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
        # Call generic fill_conf_and_schema
        cfg = super().fill_conf_and_schema(cfg)

        # Give the default value if the required element
        # is not in the configuration
        if "ranges" not in cfg:
            cfg["ranges"] = self._RANGES

        # Add subclass parameter to the default schema
        self.schema["ranges"] = list
        self.check_ranges(cfg)
        return cfg

    @staticmethod
    def check_ranges(cfg: dict) -> None:
        """
        Verify users configuration for ranges in slope classification
        :param cfg: slope configuration
        :type cfg: dict
        :return: None
        """
        ranges_dict = cfg["ranges"]
        if not all(
            isinstance(values, int) or (ranges_dict is list)
            for values in ranges_dict
        ):
            logging.error("Ranges must be a list of int")
            sys.exit(1)

    def _create_labelled_map(self):
        """
        Create the labelled map and save it if necessary
        :return: None
        """

        # transform 'ranges' to 'classes'
        self.classes: collections.OrderedDict = self._generate_classes(
            self.ranges
        )

        # create slope maps of ref and sec
        self._create_slope_map_datasets(self.dem)

    def _create_slope_map_datasets(self, dem: xr.Dataset):
        """
        Create slope map datasets

        :param dem: input dem
        :type dem:    xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :return: None
        """
        # Classify slope
        dict_slope = {"ref_slope": "ref", "sec_slope": "sec"}

        for slope_name, support in dict_slope.items():
            if slope_name in dem:
                slope_img = self.dem[slope_name].data[:, :]
                slope_dataset = create_dem(
                    slope_img,
                    transform=dem.georef_transform.data,
                    img_crs=dem.crs,
                    nodata=self.nodata,
                )
                # Create the layer map for each slope
                self._classify_slope_by_ranges(slope_dataset, support)

    @staticmethod
    def _generate_classes(ranges) -> collections.OrderedDict:
        """
        Create classes from ranges

        :param ranges: ranges
        :type ranges: List
        :return: classes
        :rtype: collections.OrderedDict
        """
        # Change the intervals into a list to make 'classes' generic
        classes = collections.OrderedDict()
        for idx, range_item in enumerate(ranges):
            if idx == len(ranges) - 1:
                key = f"[{range_item}%;inf["
            else:
                key = f"[{range_item}%;{ranges[idx + 1]}%["
            classes[key] = ranges[idx]

        return classes

    def _classify_slope_by_ranges(
        self, slope_dataset: xr.Dataset, support: str = "ref"
    ):
        """
        Create the map for each slope using the input ranges
        (value interval is transformed into 1 value (interval minimum value))

        :param slope_dataset: slope dataset
        :type slope_dataset:    xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :param support: support dem, ref or sec
        :type support: str
        :return: None
        """
        # Use radiometric ranges to classify the slope dataset
        # Initialize map
        map_img = np.ones(slope_dataset["image"].data.shape) * self.nodata
        # For each radiometric range, add the slope values that are within
        # the interval to the map_img
        for idx, _ in enumerate(self.ranges):
            # If it is the last range, do not check if smaller than next range
            if idx == len(self.ranges) - 1:
                map_img[
                    np.where(
                        (~np.isnan(slope_dataset["image"].data))
                        * (slope_dataset["image"].data >= self.ranges[idx])
                    )
                ] = self.ranges[idx]
            else:
                map_img[
                    np.where(
                        (~np.isnan(slope_dataset["image"].data))
                        * (slope_dataset["image"].data >= self.ranges[idx])
                        & (slope_dataset["image"].data < self.ranges[idx + 1])
                    )
                ] = self.ranges[idx]
        # Store map_image
        self.map_image[support] = map_img
        # If _output_dir is set, create map_dataset and save
        if self._output_dir:
            self.save_map_img(map_img, support)
