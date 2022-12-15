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
Mainly contains the ClassificationLayer class.
A classification_layer defines a way to classify the DEMs alti differences.
"""
import collections

# Standard imports
import copy
import logging
import os
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple, Union

# Third party imports
import numpy as np
import xarray as xr
from json_checker import Checker, Or

from demcompare.dem_tools import DEFAULT_NODATA, create_dem, save_dem
from demcompare.metric import Metric

# DEMcompare imports
from demcompare.output_tree_design import get_out_dir

from ..stats_dataset import StatsDataset


# pylint:disable=too-many-instance-attributes
class ClassificationLayerTemplate(metaclass=ABCMeta):
    """
    ClassificationLayer class
    A classification_layer defines a way to classify the DEM
    for the stats computation.
    """

    class NotEnoughDataToClassificationLayerError(Exception):
        """Define a NotEnoughDataToClassificationLayerError Exception"""

    class LackOfClassificationLayerDataError(Exception):
        """Define a LackOfClassificationLayerDataError Exception"""

        def __init__(self):
            super().__init__()
            logging.error(
                "At least one classification_layer"
                " support must be provided "
                "(shall it be linked to the"
                " reference DEM or the dem to align one). "
                "Use the 'ref' and/or 'sec' keys respectively."
            )

    def __init__(
        self,
        name: str,
        classification_layer_kind: str,
        cfg: Dict,
        dem: xr.Dataset = None,
    ):
        """
        Initialization of a classification_layer object

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

        # Classification_layer name
        self.name = name
        # Classification layer type
        self.type_layer = classification_layer_kind
        # Init cfg schema
        self.schema: Dict = None
        # Init classes
        self.classes: collections.OrderedDict = None
        # Dem to be classified
        self.dem: xr.Dataset = dem
        # Fill configuration file
        self.cfg: Dict = self.fill_conf_and_schema(cfg)
        # Check and update configuration file
        self.cfg = self.check_conf(self.cfg)
        # Nodata value
        self.nodata: Union[float, int] = self.cfg["nodata"]
        # Remove outliers
        self.remove_outliers: bool = self.cfg["remove_outliers"]
        # Output directory
        self._output_dir: Union[str, None] = self.cfg["output_dir"]
        # Output directory for plots
        self._plots_dir: Union[str, None] = None
        # Output directory for stats
        self._stats_dir: Union[str, None] = None
        # Create output dir (where to store classification_layer results & data)
        if self._output_dir:
            # Create stats dir
            self._stats_dir = os.path.join(
                self._output_dir, get_out_dir("_stats_dir"), self.name
            )
            os.makedirs(self._stats_dir, exist_ok=True)

        # Init labelled map data
        self.map_image: Dict = {"ref": None, "sec": None}
        # Init sets masks dict
        self.classes_masks: Dict = {"ref": [], "sec": []}

        # Init outliers free mask
        self.outliers_free_mask: np.ndarray = None

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

        # Give the default value if the required element
        # is not in the configuration
        if "nodata" not in cfg:
            cfg["nodata"] = DEFAULT_NODATA
        if "remove_outliers" in cfg:
            cfg["remove_outliers"] = cfg["remove_outliers"] == "True"
        else:
            cfg["remove_outliers"] = False
        if "output_dir" not in cfg:
            cfg["output_dir"] = None
        # Configuration schema
        self.schema = {
            "type": Or("slope", "segmentation", "global", "fusion"),
            "remove_outliers": bool,
            "output_dir": Or(str, None),
            "nodata": Or(int, float),
            "metrics": list,
        }
        return cfg

    def check_conf(self, cfg: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if the config is correct according
        to the class configuration schema

        raises CheckerError if configuration invalid.

        :param cfg: coregistration configuration
        :type cfg: Dict[str, Any]
        :return cfg: coregistration configuration updated
        :rtype: Dict[str, Any]
        """

        checker = Checker(self.schema)
        cfg = checker.validate(cfg)
        return cfg

    def compute_classif_stats(
        self,
        data: xr.Dataset,
        stats_dataset: StatsDataset,
        metrics: List[Union[dict, str]] = None,
    ):
        """
        Stats are computed based on the classification layers, which define
        classes of pixels that classify divide the input image.
        Stats are computed on each classes separately.

        Input dems can be classified by two maps belonging
        to the same classification_layer.
        If both maps exist,
        then this method produces stats based on 3 modes:

        standard mode, intersection mode: where only alti
        errors values associated with intersection classes
        between both classified images are used,
        exclusion mode: the intersection complementary one

        :param data: array to compute stats from
        :type data:    xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
                - classification_layer_masks : 3D (row, col, indicator)
                  xr.DataArray
        :param stats_dataset: StatsDataset object
        :type stats_dataset: StatsDataset
        :param metrics: metrics to be computed
        :type metrics: List[Union[dict, str]]
        :return: stats, masks, names per mode
        :rtype: List, List List
        """
        # Get outliers free mask (array of True where value is no outlier)
        self.outliers_free_mask = self._get_outliers_free_mask(
            copy.deepcopy(data["image"].data), data.attrs["nodata"]
        )
        # Get mode masks and names
        mode_masks, mode_names = self._create_mode_masks(data)

        # Compute stats for each mode
        for mode_idx, mode_name in enumerate(mode_names):
            # Compute stats for all classes of a single mode
            # and add them to the stats_dataset object
            stats_dataset = self._compute_mode_stats(
                copy.deepcopy(data["image"].data),
                stats_dataset,
                mode_mask=mode_masks[mode_idx],
                mode_name=mode_name,
                metrics=metrics,
            )

        # Save stats as plots, csv and json and do so for each mode
        if self._output_dir:
            stats_dataset.save_as_csv_and_json(self.name, self._stats_dir)

    def create_metrics(
        self, input_metrics: List[Union[dict, str]] = None
    ) -> Tuple[Dict[str, Metric], List[bool]]:
        """
        Create metric objects and remove_outliers_list

        :param input_metrics: list of input metrics
        :type input_metrics: List of Dict and str
        :return: Dict with metric objects and list
            with outlier handling per metric
        :rtype: Tuple[Dict[str, Metric], List[bool]]
        """
        # Metric objects dictionary
        metrics = {}
        # Initialize remove outliers list
        remove_outliers_list = []
        for input_metric in input_metrics:
            # The input metric can be a dictionary or a str
            # If dict, metric will have type and params
            if isinstance(input_metric, dict):
                # Get metric name
                name = list(input_metric.keys())[0]
                # Get metric params
                params = input_metric[name]
                # If outliers were specified for this metric
                if "remove_outliers" in params:
                    remove_outliers_list.append(
                        params["remove_outliers"] == "True"
                    )
                    # copy the params and suppress the
                    # remove_outliers parameter as it is
                    # not part of the metric class
                    tmp_params = copy.deepcopy(params)
                    tmp_params.pop("remove_outliers")
                    # Create metric object and add it to the dictionary
                    metrics[name] = Metric(name, tmp_params)
                else:
                    # If outliers were not specified for this metric,
                    # add the value defined for the whole classification layer
                    remove_outliers_list.append(self.remove_outliers)
                    # Create metric object and add it to the dictionary
                    metrics[name] = Metric(name, params)
            else:
                # Create the metric object and add it to the dictionary
                metrics[input_metric] = Metric(input_metric)
                # Add the outliers value defined for the whole classif layer
                remove_outliers_list.append(self.remove_outliers)
        return metrics, remove_outliers_list

    def _get_outliers_free_mask(
        self, array: np.ndarray, nodata_value: Union[int, None] = None
    ) -> np.ndarray:
        """
        Get outliers free mask (array of True where value is no outlier) with
        values outside (mu + 3 sigma) and (mu - 3 sigma).
        Nan and nodata_value are not considered in mu and sigma computation.

        :param array: input array to get the mask from
        :type array: np.ndarray
        :param nodata_value: no data value considered. Default: None
        :type nodata_value: int or None
        :return: outliers free mask (array of True where value is no outlier)
        :rtype: np.ndarray
        """
        # Get nonan and nodata mask
        nodata_free_mask = self._get_nonan_mask(array, nodata_value)
        # Apply the nonan and nodata mask to the input array
        array_without_nan = array[np.where(nodata_free_mask)]
        # Compute mean and std of the input array
        mu = np.mean(array_without_nan)
        sigma = np.std(array_without_nan)
        # Compute the outliers free mask on the input array
        return np.apply_along_axis(
            lambda x: (x > mu - 3 * sigma) * (x < mu + 3 * sigma), 0, array
        )

    def _create_mode_masks(self, alti_map: xr.Dataset):
        """
        Compute Masks for every required modes :

        the 'standard' mode: nan free, nodata free mask

        the 'intersection' mode:
        which is the 'standard' mode where only the pixels
        for which both sets (sec and ref) are intersection

        the 'exclusion' mode:
        which is 'intersection' complementary

        Note that 'intersection'
        and 'exclusion' mode masks
        can only be computed if len(_sets_masks)==2

        :param alti_map: alti differences
        :type alti_map:    xr.DataSet containing :

                - image : 2D (row, col) xr.DataArray float32
                - georef_transform: 1D (trans_len) xr.DataArray
        :return: list of masks, associated modes, and error_img read as array
        :rtype: List[np.ndarray]
        """

        mode_names = []
        mode_masks = []

        # Starting with the 'standard' mask
        mode_names.append("standard")
        # Remove alti_map nodata and nan indices
        mode_masks.append(
            self._get_nonan_mask(
                alti_map["image"].data, alti_map.attrs["nodata"]
            )
        )
        # If both sets masks have been defined, compute
        # the cross classification (intersection & exclusion) masks
        if self.classes_masks["ref"] and self.classes_masks["sec"]:
            mode_names.append("intersection")
            # Combine pairs of sets together
            # (meaning pixels belonging for the same set on
            # boths ref and sec dems)
            # for each single class / set,
            #    we know which pixels are intersection between both
            #    classification_layer_masks
            # combine_sets[0].shape[0] = number of sets (classes)
            # combine_sets[0].shape[1] = number of pixels inside a single DEM
            combine_sets = np.array(
                [
                    self.classes_masks["ref"][set_idx][:]
                    == self.classes_masks["sec"][set_idx][:]
                    for set_idx in range(0, len(self.classes_masks["ref"]))
                ]
            )
            coherent_mask = np.all(combine_sets, axis=0)
            mode_masks.append(mode_masks[0] * coherent_mask)

            # Add the exclusion one as the intersection complementary
            mode_names.append("exclusion")
            mode_masks.append(mode_masks[0] * ~coherent_mask)

        return mode_masks, mode_names

    @staticmethod
    def _get_nonan_mask(
        array: np.ndarray, nodata_value: Union[int, None] = None
    ) -> np.ndarray:
        """
        Get no data and nan mask value

        :param array: input array to get the mask from
        :type array: np.ndarray
        :param nodata_value: no data value considered. Default: None
        :type nodata_value: int or None
        :return: nan and nodata_value if exists mask on array.
        :rtype: np.ndarray
        """
        # If no nodata value is specified, just detect nan values
        if nodata_value is None:
            return np.apply_along_axis(lambda x: (~np.isnan(x)), 0, array)
        # If nodata value is specified, detect both nan and nodata values
        return np.apply_along_axis(
            lambda x: (~np.isnan(x)) * (x != nodata_value), 0, array
        )

    def _compute_mode_stats(
        self,
        dz_values: np.ndarray,
        stats_dataset: StatsDataset,
        mode_mask: np.ndarray = None,
        mode_name: str = None,
        metrics: List[Union[dict, str]] = None,
    ) -> StatsDataset:
        """
        Get stats for a specific mode

        :param dz_values: alti map
        :type dz_values: np.ndarray
        :param stats_dataset: StatsDataset object
        :type stats_dataset: StatsDataset
        :param mode_mask: boolean mask with
                True values for pixels to use
        :type mode_mask: List[bool]
        :param mode_name: mode name
        :type mode_name: str
        :param metrics: metrics to be computed
        :type metrics: List[Union[dict, str]]
        :return: StatsDataset with computed metrics (set_name, nbpts,
                 %(out_of_all_pts), max, min, mean, std, rmse, ...)
        :rtype: StatsDataset
        """
        # Initialize stats_list
        stats_list = []
        # Total number of points
        nb_total_points = dz_values.size

        # If no metrics have been specified, consider
        # the given cfg metrics
        if metrics is None:
            metrics = self.cfg["metrics"]

        logging.debug("Computing mode %s, metrics: %s", mode_name, metrics)

        # Consider either the "ref" or "sec" classes masks.
        # If both exist, "ref" is considered
        if self.classes_masks["ref"]:
            class_masks = self.classes_masks["ref"]
        else:
            class_masks = self.classes_masks["sec"]

        # Compute stats for each class
        if class_masks is not None:
            for idx, (class_name, class_item) in enumerate(
                self.classes.items()
            ):
                # Class altitude values
                class_alti_values = dz_values[
                    np.where((class_masks[idx] * mode_mask))
                ]
                # Class outliers free mask
                class_outliers_free_mask = (
                    class_masks[idx] * mode_mask * self.outliers_free_mask
                )
                # Class outliers free altitude values
                class_outliers_free_alti_values = dz_values[
                    np.where(class_outliers_free_mask)
                ]
                # Do stats computation and obtain class_stats dictionary
                class_stats = self.stats_computation(
                    class_alti_values, class_outliers_free_alti_values, metrics
                )
                # Masks altitude values not corresponding to the
                # class and its mode
                # and save it as the dz_values stats
                class_dz_values = copy.deepcopy(dz_values)
                class_dz_values[
                    np.where((class_masks[idx] * mode_mask) == 0)
                ] = np.nan
                class_stats["dz_values"] = class_dz_values
                # Add nbpts value
                class_stats["nbpts"] = class_alti_values.size
                # Add class name
                class_stats["class_name"] = class_name + ":" + str(class_item)
                # Add percent_valid_points value
                class_stats["percent_valid_points"] = round(
                    (100 * class_alti_values.size / float(nb_total_points)),
                    5,
                )
                # Add the class_stats dictionary to the stats_list
                # Need to copy, otherwise the array dz_values is overwritten
                stats_list.append(copy.deepcopy(class_stats))
        # Add the obtained stats on the stats_dataset object
        stats_dataset.add_classif_layer_and_mode_stats(
            classif_name=self.name, input_stats=stats_list, mode_name=mode_name
        )
        return stats_dataset

    def _create_class_masks(self):
        """
        Returns a list of masks, by class.
        Each masks indicates which pixels belong
        to the class.

        :return: None
        """
        # For each support ("ref" and "sec")
        for support in self.classes_masks:
            # If the support is present
            if self.map_image[support] is not None:
                # Initialize support map
                support_masks = []
                # Get map_image to be classified by classes
                img_to_classify = self.map_image[support]
                # For each class on the classification layer
                for _, class_value in self.classes.items():
                    # Obtain the positions where the map image
                    # has the class value
                    if isinstance(class_value, list):
                        if len(class_value) == 1:
                            # transform it to value
                            class_value = class_value[0]
                    if isinstance(class_value, list):
                        class_positions = np.where(
                            np.logical_or(
                                *[
                                    np.equal(img_to_classify, label_i)
                                    for label_i in class_value
                                ]
                            )
                        )

                    else:
                        class_positions = np.where(
                            img_to_classify == class_value
                        )
                    # Initialize class mask and add class positions to True
                    mask = np.ones(img_to_classify.shape) * False
                    mask[class_positions] = True
                    # Add class mask to the support mask
                    support_masks.append(mask)
                self.classes_masks[support] = support_masks

    @abstractmethod
    def _create_labelled_map(self):
        """
        Creates labelled map
        :return: None
        """

    def stats_computation(
        self,
        data: np.ndarray,
        outliers_free_data: np.ndarray,
        input_metrics: List[Union[str, Dict]] = None,
    ) -> Dict:
        """
        Compute stats for a specific array

        :param data: input data
        :type data: np.ndarray
        :param outliers_free_data: input outliers_free_data
        :type outliers_free_data: np.ndarray
        :param input_metrics: input metrics to use
        :type input_metrics: List[Union[str, Dict]]
        :return: dict with computed metric values
        :rtype: Dict
        """
        # Create metrics and outliers indicator list
        metrics, remove_outliers_list = self.create_metrics(input_metrics)
        # Initialize metric results dict
        metric_results: Dict = {}
        # Iterate over each metrics
        for idx, (metric_name, metric_object) in enumerate(metrics.items()):
            # Choose array according to outliers configuration of the metric
            if remove_outliers_list[idx]:
                array = outliers_free_data
            else:
                array = data
            if array.size:
                # Format output list according to the metric type
                # Round the float results
                if metric_object.type == "scalar":
                    computed_metric = metric_object.compute_metric(array)
                    metric_results[metric_name] = round(
                        float(computed_metric), 5
                    )
                elif metric_object.type == "vector" and not np.all(
                    np.round(array, decimals=6) == 0
                ):
                    computed_metric = metric_object.compute_metric(array)
                    for idx_vec, _ in enumerate(computed_metric[0]):
                        computed_metric[0][idx_vec] = round(
                            float(computed_metric[0][idx_vec]), 5
                        )
                        computed_metric[1][idx_vec] = round(
                            float(computed_metric[1][idx_vec]), 5
                        )
                    metric_results[metric_name] = (
                        computed_metric[0],
                        computed_metric[1],
                    )
                elif metric_object.type == "vector" and np.all(
                    np.round(array, decimals=6) == 0
                ):
                    logging.warning(
                        "%s is not computed because reference and "
                        "second DEMs are the same",
                        metric_name,
                    )
            else:
                # If the input array is empty, the metric is np.nan
                if metric_object.type == "scalar":
                    metric_results[metric_name] = np.nan
                elif metric_object.type == "vector":
                    metric_results[metric_name] = (np.nan, np.nan)

        return metric_results

    def save_map_img(self, map_img: np.ndarray, map_support: str):
        """
        Save the classification layer map to file

        :param map_img: input data
        :type map_img: np.ndarray
        :param map_support: map support "ref" or "sec
        :type map_support: str
        :return: None
        """

        map_dataset = create_dem(
            map_img,
            self.dem.georef_transform.data,
            img_crs=self.dem.crs,
            nodata=self.nodata,
        )
        map_path = os.path.join(
            self._stats_dir, map_support + "_rectified_support_map.tif"
        )
        save_dem(map_dataset, map_path, nodata=self.nodata)
