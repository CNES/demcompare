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
Mainly contains the StatsDataset class
contains the computed stats of a pair of DEMs
for the different classification layers
"""

# Standard imports
import collections
import copy
import csv
import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import xarray as xr


class StatsDataset:
    """
    StatsDataset class

    The StatsDataset class contains a list of one
    xr.dataset per classification layer

    Each xr.Dataset contains :

    :image: 2D (row, col) input image as xarray.DataArray,

    :image_by_class: 3D (row, col; nb_classes)

        xarray.DataArray containing
        the image pixels belonging
        to each class considering the valid pixels

    :image_by_class_intersection: 3D (row, col; nb_classes)

        xarray.DataArray containing
        the image pixels belonging
        to each class considering the intersection mode

    :image_by_class_exclusion: 3D (row, col; nb_classes)

        xarray.DataArray containing
        the image pixels belonging
        to each class considering the exclusion mode

    :attributes:

                - name : name of the classification_layer. str

                - stats_by_class : dictionary containing
                  the stats per class considering the standard mode

                - stats_by_class_intersection : dictionary containing
                  the stats per class considering the intersection mode

                - stats_by_class_exclusion : dictionary containing
                  the stats per class considering the exclusion mode

    """

    def __init__(self, image: np.ndarray):
        # Dictionary with the different classification layers
        # and the modes of each layer
        self.classif_layers_and_modes: Dict = {}
        # Image map
        self.image: np.ndarray = image
        # List of xr.Dataset for each classification layer
        self.classif_layers_dataset: List[xr.Dataset] = []

    def add_classif_layer_and_mode_stats(
        self, classif_name: str, input_stats: List[Dict], mode_name: str
    ):
        """
        Add the stats of a classification layer and
        a mode to the corresponding xarray dataset

        :param classif_name: classification_layer name
        :type classif_name: str
        :param input_stats: input statistics
        :type input_stats: List[str]
        :mode_name: name of the mode (standard (no name),
          intersection, exclusion)
        :type mode_name: str
        :return: None
        """
        # If no xr.Dataset exists for the classification layer,
        # create it
        if classif_name not in self.classif_layers_and_modes:
            # Store the classification layer name on the
            # classif_layers_and_modes dictionary
            self.classif_layers_and_modes[classif_name] = {}
            self.classif_layers_and_modes[classif_name]["modes"] = []
            # Initialize the dataset
            new_dataset = xr.Dataset(
                {"image": (["row", "col"], self.image)},
                coords={
                    "row": np.arange(self.image.shape[0]),
                    "col": np.arange(self.image.shape[1]),
                },
            )
            # Add the name of the classification as an attribute
            new_dataset.attrs["name"] = classif_name
            # Add the created dataset to the classif_layers_dataset list
            self.classif_layers_dataset.append(new_dataset)

        # Image and stats indicator name
        if mode_name == "standard":
            image_indicator = "image_by_class"
            stats_indicator = "stats_by_class"
        else:
            image_indicator = "image_by_class_" + mode_name
            stats_indicator = "stats_by_class_" + mode_name

        # Add the mode of the corresponding classification layer on the
        # classif_layers_and_modes dictionary
        self.classif_layers_and_modes[classif_name]["modes"].append(mode_name)
        # Get the corresponding dataset idx
        dataset_idx = list(self.classif_layers_and_modes.keys()).index(
            classif_name
        )
        # Initialize the classification layer classes
        classes = list(np.arange(len(input_stats)))
        # Define coords, the third col is the indicator
        # with the number of classes
        coords_classification_layers = [
            self.classif_layers_dataset[dataset_idx].coords["row"],
            self.classif_layers_dataset[dataset_idx].coords["col"],
            classes,
        ]
        # Initialize the image data by class
        # Each dataset has one xr.DataArray per mode indicating
        # the image by class
        image_maps = np.full(
            (
                self.image.shape[0],
                self.image.shape[1],
                len(classes),
            ),
            np.nan,
            dtype=np.float32,
        )
        # Initialize the stats_by_class + mode_name dictionary on
        # the dataset attrs
        if (
            stats_indicator
            not in self.classif_layers_dataset[dataset_idx].attrs
        ):
            self.classif_layers_dataset[dataset_idx].attrs[stats_indicator] = {}
        # Iterate to obtain the stats per class
        for class_idx, class_stats in enumerate(input_stats):
            # Fill the alti diff of the corresponding class
            # with the input dz_values
            image_maps[:, :, class_idx] = class_stats["dz_values"]
            # Make a copy of the class_stats dictionary to make
            # temporal changes
            tmp_class_stats = copy.deepcopy(class_stats)
            # Pop the dz_values dictionary key to iterate over
            # the rest of metrics
            tmp_class_stats.pop("dz_values")
            # Scalar metrics are stored in attrs
            # of the dataset
            # Initialize the stats_by_class + mode_name +
            # class_idx dictionary
            if (
                class_idx
                not in self.classif_layers_dataset[dataset_idx].attrs[
                    stats_indicator
                ]
            ):
                self.classif_layers_dataset[dataset_idx].attrs[stats_indicator][
                    class_idx
                ] = {}
            # Add each metric on the dictionary
            for stat_name, stat_value in tmp_class_stats.items():
                self.classif_layers_dataset[dataset_idx].attrs[stats_indicator][
                    class_idx
                ][stat_name] = stat_value

        # Create and add the xr.DataArray to the dataset
        self.classif_layers_dataset[dataset_idx][
            image_indicator
        ] = xr.DataArray(
            data=image_maps,
            coords=coords_classification_layers,
            dims=["row", "col", "classes"],
        )

    def save_as_csv_and_json(
        self,
        classif_name: str,
        stats_dir: str,
    ):
        """
        Saves the classification layer's results to csv and json
        files on the stats_dir
        :param classif_name: classification_layer name
        :type classif_name: str
        :param stats_dir: output stats directory
        :type stats_dir: str
        :return: None
        """
        # Iterate over the classification modes
        for _, mode_name_item in enumerate(
            self.classif_layers_and_modes[classif_name]["modes"]
        ):
            # Get the dataset idx of the corresponding classification layer
            dataset_idx = list(self.classif_layers_and_modes.keys()).index(
                classif_name
            )
            # Get the xr.Dataset
            classif_dataset = self.classif_layers_dataset[dataset_idx]
            # Indicator name of the image and stats by class map
            if mode_name_item == "standard":
                mode_name_item = ""
            else:
                mode_name_item = "_" + mode_name_item

            stats_dict = classif_dataset.attrs[
                "stats_by_class" + mode_name_item
            ]
            scalar_metric_dict: collections.OrderedDict = (
                collections.OrderedDict()
            )
            # Add each class stats on the results dict
            for class_idx in list(stats_dict.keys()):
                scalar_metric_dict[class_idx] = {}
                scalar_metric_dict[class_idx]["Set Name"] = stats_dict[
                    class_idx
                ]["class_name"]
                # Save scalar metrics
                for metric_name, metric_stats in stats_dict[class_idx].items():
                    if isinstance(metric_stats, (float, int)):
                        scalar_metric_dict[class_idx][
                            metric_name
                        ] = metric_stats
            # Initialize the output json path
            mode_output_json_files = os.path.join(
                stats_dir, "stats_results" + mode_name_item + ".json"
            )
            # Save the results dictionary on a json file
            with open(mode_output_json_files, "w", encoding="utf8") as outfile:
                json.dump(scalar_metric_dict, outfile, indent=4)

            # Save the results into a csv file
            # - create filename
            csv_filename = os.path.join(
                os.path.splitext(mode_output_json_files)[0] + ".csv"
            )

            # - writes the results down as csv format
            with open(csv_filename, "w", encoding="utf8") as csvfile:
                fieldnames = list(scalar_metric_dict[0].keys())
                writer = csv.DictWriter(
                    csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC
                )

                writer.writeheader()
                for set_item in scalar_metric_dict:
                    writer.writerow(scalar_metric_dict[set_item])

    def get_classification_layer_names(self):
        """
        Returns the available classification layers

        :return: None
        """
        return list(self.classif_layers_and_modes.keys())

    def get_classification_layer_dataset(
        self, classification_layer: str
    ) -> xr.Dataset:
        """
        Returns the xr.Dataset corresponding to
        the input classification layer name

        :param classification_layer: classification_layer name
        :type classification_layer: str
        :return: stats dictionary
        :rtype: xr.Dataset
        """
        # Get the dataset index  and return the corresponding dataset
        idx = list(self.classif_layers_and_modes.keys()).index(
            classification_layer  # pylint:disable=consider-iterating-dictionary
        )
        return self.classif_layers_dataset[idx]

    def get_classification_layer_stats(self, classification_layer: str) -> Dict:
        """
        Returns all the stats corresponding to
        the input classification layer name

        :param classification_layer: classification_layer name
        :type classification_layer: str
        :return: stats dictionary
        :rtype: Dict
        """
        # Get the dataset index  and return the corresponding dataset
        idx = list(self.classif_layers_and_modes.keys()).index(
            classification_layer  # pylint:disable=consider-iterating-dictionary
        )
        return self.classif_layers_dataset[idx].attrs

    def get_classification_layer_metrics(
        self,
        classification_layer: str,
    ) -> List[str]:
        """
        Returns the metric names available on the
        input classification layer and mode

        :param classification_layer: classification_layer name
        :type classification_layer: str
        :return: available metric names
        :rtype: List[str]
        """
        # Get classification_layer dataset
        dataset = self.get_classification_layer_dataset(classification_layer)
        # Get available metric names
        output_metric_names = copy.deepcopy(
            list(dataset.attrs["stats_by_class"][0].keys())
        )
        # Delete class name as it is not a metric
        output_metric_names.pop(output_metric_names.index("class_name"))
        return output_metric_names

    def get_classification_layer_metric(
        self,
        classification_layer: str,
        classif_class: int = None,
        mode: str = "",
        metric: str = None,
    ) -> Union[List, Tuple[np.ndarray, np.ndarray], np.ndarray, float]:
        """
        Returns the metric corresponding to the
        input classification layer and mode

        :param classification_layer: classification_layer name
        :type classification_layer: str
        :param classif_class: classification_layer class
        :type classif_class: int
        :param mode: mode (standard (no name), intersection, exclusion)
        :type mode: str
        :param metric: metric
        :type metric: str
        :return: metric
        :rtype: Union[List,Tuple[np.ndarray, np.ndarray], np.ndarray, float]
        """
        # Get classification_layer dataset
        dataset = self.get_classification_layer_dataset(classification_layer)
        if mode in ("standard", ""):
            # Standard mode
            stats_indicator = "stats_by_class"
        else:
            stats_indicator = "stats_by_class_" + mode
        # If the class was specified, return the corresponding metric
        if isinstance(classif_class, int):
            output_metric = dataset.attrs[stats_indicator][classif_class][
                metric
            ]
        # Otherwise, return a list with the metric for each class
        else:
            output_metric = []
            # Iterate over the classes and add the metric
            # result of each class
            for _, metric_dict in dataset.attrs[stats_indicator].items():
                output_metric.append(metric_dict[metric])
        return output_metric
