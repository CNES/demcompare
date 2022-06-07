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
# pylint:disable=too-many-lines, too-many-branches
"""
Mainly contains the Partition class.
A partition defines a way to partition the DEMs alti differences.
TODO add comment FusionPartition
"""

# Standard imports
import collections
import itertools
import logging
import os
from functools import reduce
from typing import Dict, List

# Third party imports
import matplotlib
import matplotlib.pyplot as mpl_pyplot
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

# DEMcompare imports
from .dem_tools import (
    SamplingSourceParameter,
    create_dem,
    load_dem,
    reproject_dems,
    save_dem,
)
from .output_tree_design import get_out_dir
from .slope import get_slope


class NotEnoughDataToPartitionError(Exception):
    """Define a specific NotEnoughDataToPartitionError Exception"""


class Partition:
    """
    Partition class
    A partition defines a way to partition the DEMs alti differences.
    """

    # pylint: disable=too-many-instance-attributes

    class LackOfPartitionDataError(Exception):
        """Define a specific LackOfPartitionDataError Exception"""

        def __init__(self):
            super().__init__()
            logging.error(
                "At least one partition support must be provided "
                "(shall it be linked to the reference DSM or the slave one). "
                "Use the 'ref' and/or 'dsm' keys respectively."
            )

    # Only kind of partition supported
    # TODO see if two classes will not be better
    type = ["to_be_classification_layers", "classification_layers"]

    # default no data value
    nodata = -32768.0

    #
    # Initialization
    #
    def __init__(  # pylint:disable=too-many-arguments
        self,
        name: str,
        partition_kind: str,
        coreg_dsm: xr.Dataset,
        coreg_ref: xr.Dataset,
        output_dir: str,
        sampling_source: str = SamplingSourceParameter.DEM_TO_ALIGN.value,
        # Disable for unreformable line_too_long
        geo_ref: bool = True,
        dec_ref_path: str = None,
        dec_dem_path: str = None,
        init_disp_x: int = None,
        init_disp_y: int = None,
        dx: float = 0.0,
        dy: float = 0.0,
        **cfg_layer: Dict
    ):
        """
        Initialization of a partition object

        :param name: partition name
        :type name: str
        :param partition_kind: partition kind
        :type partition_kind: str
        :param coreg_dsm: coregistered dsm
        :type coreg_dsm: xr.Dataset
        :param coreg_ref: coregistered ref
        :type coreg_ref: xr.Dataset
        :param output_dir: output directory
        :type output_dir: str
        :param sampling_source: sampling source for reprojection
        :type sampling_source: str
        :param geo_ref: georeference
        :type geo_ref: bool
        :param dec_ref_path: decorelated ref path
        :type dec_ref_path: str or None
        :param dec_dem_path: decorelated dem path
        :type dec_dem_path: str or None
        :param init_disp_x: intiial disparity x
        :type init_disp_x: int or None
        :param init_disp_y: initial disparity y
        :type init_disp_y: int or None
        :param dx: Nuth offset x
        :type dx: float or None
        :param dy: Nuth offset y
        :type dy: float or None
        :return: None
        """

        # Sanity check
        if partition_kind in self.type:
            self._type_layer = partition_kind
        else:
            logging.error(
                "Unsupported partition kind {}. \
                Try one of the following {}".format(
                    partition_kind, self.type
                )
            )
            raise KeyError

        # Get partition name
        self._name = name

        # Init classes
        self._classes = None

        # Create output dir (where to store partition results & data)
        self._output_dir = output_dir
        self.create_output_dir()

        # Store coreg path (TODO why?)
        self.coreg_path = {"ref": coreg_ref, "dsm": coreg_dsm}
        self._coreg_shape = coreg_ref["image"].data.shape

        # Get coregistration offsets
        self.dx = dx
        # Negative because the cfg file saves the negative of dy
        self.dy = -dy
        self.init_disp_x = init_disp_x
        self.init_disp_y = init_disp_y
        self.sampling_source = sampling_source
        # Get uncoregistred ref and dem path
        self.dec_ref_path = dec_ref_path
        self.dec_dem_path = dec_dem_path

        # Init input data path
        self.ref_path = ""
        self.dsm_path = ""

        # Init labelled map data
        self.reproject_path = {"ref": None, "dsm": None}
        self.map_path = {"ref": None, "dsm": None}

        # Init sets attributes
        self._sets_indexes = {"ref": [], "dsm": []}
        self._sets_names = None
        self._sets_labels = None
        self._sets_masks = None
        self._sets_colors = None

        # Georef set
        self.geo_ref = geo_ref

        # Create partition (labelled map with associated sets)
        self._create_partition_sets(**cfg_layer)

        logging.debug("Partition created as: {}".format(self))

    def _create_partition_sets(self, **cfg_layer):
        """
        Create partition sets
        :param cfg_layer: cfg
        :type cfg_layer: dict
        :return: None
        """
        if self.name == "global":
            # create default partition
            # no sets needed
            self._create_default_partition()
        else:
            # create labelled map to partition from
            self._create_labelled_map(**cfg_layer)

            # fill sets
            self._fill_sets_attributes()

    def _create_default_partition(self):
        """
        Create default partition.

        :return: None
        """
        self._sets_masks = [
            ~(
                np.isnan(self.coreg_path["dsm"]["image"].data)
                * np.isnan(self.coreg_path["ref"]["image"].data)
            )
        ]
        self._sets_colors = None

    #
    # Getters and setters
    #
    @property
    def out_dir(self):
        return self._output_dir

    @property
    def stats_dir(self):
        return os.path.join(self.out_dir, get_out_dir("stats_dir"), self._name)

    @property
    def histograms_dir(self):
        return os.path.join(
            self.out_dir, get_out_dir("histograms_dir"), self._name
        )

    @property
    def plots_dir(self):
        return os.path.join(
            self.out_dir, get_out_dir("snapshots_dir"), self._name
        )

    @property
    def stats_mode_json(self):
        # {'standard': 'chemin_stats_standard.json',
        #  'coherent' etc.}
        return self._stats_mode_json_dict

    @stats_mode_json.setter
    def stats_mode_json(self, mode_json_dict):
        self._stats_mode_json_dict = mode_json_dict

    @property
    def coreg_shape(self):
        return self._coreg_shape

    @property
    def name(self):
        return self._name

    @property
    def type_layer(self):
        return self._type_layer

    @property
    def classes(self):
        return self._classes

    @property
    def sets_names(self):
        return self._sets_names

    @property
    def sets_labels(self):
        return self._sets_labels

    @property
    def sets_colors(self):
        return self._sets_colors

    @property
    def sets_indexes_ref(self):
        return self._sets_indexes["ref"]

    @property
    def sets_indexes_dsm(self):
        return self._sets_indexes["dsm"]

    @property
    def stats_results(self) -> Dict:
        """
        Return stats results of partition.

        :return: stats_results
        :type stats_results: Dict
        """
        stats_results = {}
        #
        # Mode standard
        #
        stats_results["standard"] = {"Ref_support": None, "DSM_support": None}
        if self.ref_path or self.map_path["ref"]:
            stats_results["standard"]["Ref_support"] = {
                "nodata": self.nodata,
                "path": self.map_path["ref"],
            }
        if self.dsm_path or self.map_path["dsm"]:
            stats_results["standard"]["DSM_support"] = {
                "nodata": self.nodata,
                "path": self.map_path["dsm"],
            }
        if (self.ref_path or self.map_path["ref"]) and (
            self.dsm_path or self.map_path["dsm"]
        ):
            #
            # Mode coherent
            #
            stats_results["coherent-classification"] = {
                "Ref_support": None,
                "DSM_support": None,
            }
            # TODO fill coherent with corresponding maps
            #
            # Mode incoherent
            #
            stats_results["incoherent-classification"] = {
                "Ref_support": None,
                "DSM_support": None,
            }
            # TODO fill incoherent with corresponding maps
        return stats_results

    @property
    def sets_masks(self) -> List[List[np.ndarray]]:
        """
        Set masks for partition.

        :return: masks set
        :rtype: List[List[np.ndarray]]
        """
        if self._sets_masks is None:
            all_masks = []
            ref_masks = []
            dsm_masks = []
            if self.sets_indexes_ref:
                for label_idx in range(len(self.sets_labels)):
                    ref_masks.append(np.ones(self.coreg_shape) * False)
                    ref_masks[label_idx][
                        self.sets_indexes_ref[label_idx]
                    ] = True
                all_masks.append(ref_masks)
            if self.sets_indexes_dsm:
                for label_idx in range(len(self.sets_labels)):
                    dsm_masks.append(np.ones(self.coreg_shape) * False)
                    dsm_masks[label_idx][
                        self.sets_indexes_dsm[label_idx]
                    ] = True
                all_masks.append(dsm_masks)

            self._sets_masks = all_masks

        return self._sets_masks

    def __repr__(self):
        return "\n".join(
            [
                "",
                "----",
                "| Partition `{}` of type : {}".format(
                    self.name, self.type_layer
                ),
                "| - path to REF input: ",
                "|\t{}".format(self.ref_path),
                "|   whose labeled & coregistered version is ",
                "|\t{}".format(self.map_path["ref"]),
                "| - path to DSM input:",
                "|\t{}".format(self.dsm_path),
                "|   whose labeled & coregistered version is ",
                "|\t{}".format(self.map_path["dsm"]),
                "----",
            ]
        )

    def generate_classes(self, ranges) -> collections.OrderedDict:
        """
        Create classes from ranges

        :param ranges: ranges
        :type ranges: List
        :return: classes
        :rtype: collections.OrderedDict
        """
        # change the intervals into a list to make 'classes' generic
        classes = collections.OrderedDict()
        for idx, range_item in enumerate(ranges):
            if idx == len(ranges) - 1:
                if self.name == "slope":
                    key = "[{}%;inf[".format(range_item)
                else:
                    key = "[{};inf[".format(range_item)
            else:
                if self.name == "slope":
                    key = "[{}%;{}%[".format(range_item, ranges[idx + 1])
                else:
                    key = "[{};{}[".format(range_item, ranges[idx + 1])
            classes[key] = ranges[idx]

        return classes

    def create_output_dir(self):
        """
        Create folder stats results
        :return: None
        """
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.histograms_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def create_slope(self, coreg_dsm, coreg_ref):
        """
        Create slope if not exist

        :param coreg_dsm: input coregistered DSM
        :type coreg_ds: xr.Dataset
        :param coreg_ref: input coregistered REF
        :type coreg_ref: xr.Dataset
        :return: None
        """
        # Compute ref slope
        self.ref_path = os.path.join(self.stats_dir, "Ref_support.tif")
        slope_ref = get_slope(coreg_ref, degree=False)
        slope_ref_dataset = create_dem(
            slope_ref,
            transform=coreg_ref.georef_transform.data,
            img_crs=coreg_ref.crs,
            no_data=self.nodata,
        )
        save_dem(slope_ref_dataset, self.ref_path)

        # Compute dsm slope
        self.dsm_path = os.path.join(self.stats_dir, "DSM_support.tif")
        slope_dsm = get_slope(coreg_dsm, degree=False)
        slope_dsm_georaster = create_dem(
            slope_dsm,
            transform=coreg_dsm.georef_transform.data,
            img_crs=coreg_dsm.crs,
            no_data=self.nodata,
        )
        save_dem(slope_dsm_georaster, self.dsm_path)

    def create_map(self, slope_img: str, type_slope: str):
        """
        Create the map for each slope
        (value interval is transformed into 1 value (interval minimum value)

        :param slope_img: slope image path
        :type slope_img: str
        :param type_slope: type of slope : 'ref' or 'dsm'
        :type type_slope: str
        :return: None
        """
        # use radiometric ranges to classify
        slope = load_dem(slope_img)
        rad_range = list(self.classes.values())
        map_img = create_dem(
            np.ones(slope["image"].data.shape) * self.nodata,
            transform=slope.georef_transform.data,
            img_crs=slope.crs,
            no_data=self.nodata,
        )
        map_img = save_dem(map_img, self.ref_path)

        for idx, _ in enumerate(rad_range):
            if idx == len(rad_range) - 1:
                map_img["image"].data[
                    np.where(
                        (~np.isnan(slope["image"].data))
                        * (slope["image"].data >= rad_range[idx])
                    )
                ] = rad_range[idx]
            else:
                map_img["image"].data[
                    np.where(
                        (~np.isnan(slope["image"].data))
                        * (slope["image"].data >= rad_range[idx])
                        & (slope["image"].data < rad_range[idx + 1])
                    )
                ] = rad_range[idx]

        self.map_path[type_slope] = os.path.join(
            self.stats_dir, type_slope + "_support_map.tif"
        )
        save_dem(map_img, self.map_path[type_slope])

    def _create_set_indices(self):
        """
        Returns a list of numpy.where, by class.
        Each element defines a set.
        The sets partition / classify the image.
        Each numpy.where contains the coordinates
        of the sets of the class.
        Create list of coordinates arrays :
        -> self.sets_indices = [(label_name,
        np.where(...)),... label_name, np.where(...))] ,

        :return: None
        """
        dsm_supports = ["ref", "dsm"]
        sets_indices = {support: None for support in dsm_supports}
        for support in dsm_supports:
            sets_indices[support] = []
            if self.reproject_path[support]:
                img_to_classify = load_dem(self.reproject_path[support])[
                    "image"
                ].data

                # calculate sets_indices of partition
                for class_name, class_value in self.classes.items():
                    if isinstance(class_value, list):
                        if len(class_value) == 1:
                            # transform it to value
                            class_value = class_value[0]
                    if isinstance(class_value, list):
                        elm = (
                            class_name,
                            np.where(
                                np.logical_or(
                                    *[
                                        np.equal(img_to_classify, label_i)
                                        for label_i in class_value
                                    ]
                                )
                            ),
                        )
                    else:
                        elm = (
                            class_name,
                            np.where(img_to_classify == class_value),
                        )
                    sets_indices[support].append(elm)

        return sets_indices

    def _fill_sets_names_labels(self):
        """
        Fills the labels of the sets names
        :return:
        """
        # fill sets_labels & sets_names
        if self.name == "slope":
            self._sets_names = list(self.classes.keys())
            # Slope labels are historically customized
            self._sets_labels = [
                r"$\nabla$ > {}%".format(self.classes[set_name])
                if set_name.endswith("inf[")
                else r"$\nabla \in$ {}".format(set_name)
                for set_name in self._sets_names
            ]
        else:
            self._sets_labels = list(self.classes.keys())
            self._sets_names = [
                "{}:{}".format(key, value)
                for key, value in self.classes.items()
            ]
            self._sets_names = [
                name.replace(",", ";") for name in self._sets_names
            ]

    def _fill_sets_attributes(self):
        """
        Fills the sets attributes

        :return:
        """
        self._fill_sets_names_labels()

        # fill sets_colors
        self._sets_colors = (
            np.multiply(get_color(len(self.sets_names)), 255) / 255
        )
        # fill sets_indexes
        tuples_of_labels_and_indexes = self._create_set_indices()
        if tuples_of_labels_and_indexes["ref"]:
            self._sets_indexes["ref"] = [
                item[1] for item in tuples_of_labels_and_indexes["ref"]
            ]
        if tuples_of_labels_and_indexes["dsm"]:
            self._sets_indexes["dsm"] = [
                item[1] for item in tuples_of_labels_and_indexes["dsm"]
            ]

    def _create_labelled_map(self, **cfg_layer):
        """
        Creates labelled map
        :param cfg_layer: cfg
        :type cfg_layer: dict
        :return: None
        """
        # Store classes (TODO why?)
        self._classes = {}
        if "classes" in cfg_layer:
            self._classes = collections.OrderedDict(cfg_layer["classes"])
        elif "ranges" in cfg_layer:
            # transform 'ranges' to 'classes'
            self._classes = self.generate_classes(cfg_layer["ranges"])
        else:
            logging.error(
                "Neither classes nor ranges \
                where given as input sets to partition the stats"
            )
            raise KeyError

        # Store path to initial layer
        if "ref" in cfg_layer:
            self.ref_path = cfg_layer["ref"]
        if "dsm" in cfg_layer:
            self.dsm_path = cfg_layer["dsm"]
        if ("ref" not in cfg_layer) and ("dsm" not in cfg_layer):
            raise self.LackOfPartitionDataError
        if (not self.ref_path) and (not self.dsm_path):
            if self.type_layer == "classification_layers":
                raise self.LackOfPartitionDataError
            # else
            if self.name != "slope":
                raise self.LackOfPartitionDataError
            # else
            # create slope : ref and dsm
            self.create_slope(self.coreg_path["dsm"], self.coreg_path["ref"])

        # Create the layer map
        if self.type_layer == "to_be_classification_layers":
            # if the partition is not yet a labelled map, then make it so
            if self.ref_path:
                self.create_map(self.ref_path, "ref")
            if self.dsm_path:
                self.create_map(self.dsm_path, "dsm")
        elif self.type_layer == "classification_layers":
            if "ref" in cfg_layer:
                self.map_path["ref"] = cfg_layer["ref"]
            if "dsm" in cfg_layer:
                self.map_path["dsm"] = cfg_layer["dsm"]

        # Reproj the layer map
        self.rectify_map()

    def rectify_map(self):
        """
        Rectify the layer maps according to coreg dsm and coreg ref
        (which are coregistered together)

        """
        # TODO : make distinction when in presence of :
        #  - other coregistration modes
        #  - other Nuth et Kaab implementations
        #  - other coregistration algorithms
        for map_name, map_path in self.map_path.items():
            if map_path:
                if self.geo_ref:
                    # If to_be_classification_layers, the map has already
                    # been computed with coreg dsm and coreg ref
                    if self._type_layer == "to_be_classification_layers":
                        map_img = load_dem(map_path)
                        self.reproject_path[map_name] = os.path.join(
                            self.stats_dir, map_name + "_support_map_rectif.tif"
                        )
                        save_dem(map_img, self.reproject_path[map_name])
                    # If classification_layers, we need to reproject and apply
                    # the nuth et kaab offsets for the layer to be coregistered
                    # with coreg dsm and coreg ref
                    if self._type_layer == "classification_layers":

                        if map_name == "dsm":
                            ref_orig = load_dem(self.dec_ref_path)
                            map_dataset = load_dem(map_path)
                            (
                                rectified_map_dataset,
                                ref,  # pylint:disable=unused-variable
                                adaptation_factor,
                            ) = reproject_dems(
                                map_dataset,
                                ref_orig,
                                self.init_disp_x,
                                self.init_disp_y,
                                self.sampling_source,
                            )

                            rectified_map = rectified_map_dataset["image"].data

                        elif map_name == "ref":
                            map_dataset = load_dem(map_path)
                            dem_orig = load_dem(self.dec_dem_path)
                            (
                                dem,
                                rectified_map_dataset,
                                adaptation_factor,
                            ) = reproject_dems(
                                dem_orig,
                                map_dataset,
                                self.init_disp_x,
                                self.init_disp_y,
                                self.sampling_source,
                            )
                            # Adapt offset to the reprojected ref
                            # Divide since we want to adapt the offset
                            # to the reference resolution
                            factor_x, factor_y = adaptation_factor
                            self.dx = self.dx / factor_x
                            self.dy = self.dy / factor_y
                            # Compute interpolation for the rectified map to be
                            # at the same grid as interm_coreg_REF.tif
                            xgrid = np.arange(dem["image"].data.shape[1])
                            ygrid = np.arange(dem["image"].data.shape[0])
                            nan_maskval = np.isnan(
                                rectified_map_dataset["image"].data
                            )
                            dsm_from_filled = np.where(
                                nan_maskval,
                                -9999,
                                rectified_map_dataset["image"].data,
                            )
                            spline_1 = RectBivariateSpline(
                                ygrid, xgrid, dsm_from_filled, kx=1, ky=1
                            )
                            spline_2 = RectBivariateSpline(
                                ygrid, xgrid, nan_maskval, kx=1, ky=1
                            )
                            rectified_map = spline_1(
                                ygrid - self.dy, xgrid + self.dx
                            )
                            nanval_new = spline_2(
                                ygrid - self.dy, xgrid + self.dx
                            )
                            rectified_map[nanval_new != 0] = np.nan

                        # Update map with Nuth et kaab offsets
                        map_img = rectified_map_dataset.copy()

                        if self.dx >= 0:
                            rectified_map = rectified_map[
                                :,
                                0 : map_img["image"].data.shape[1]
                                - int(np.ceil(self.dx)),
                            ]
                        else:
                            rectified_map = rectified_map[
                                :,
                                int(np.floor(-self.dx)) : map_img[
                                    "image"
                                ].data.shape[1],
                            ]
                        if -self.dy >= 0:
                            rectified_map = rectified_map[
                                0 : map_img["image"].data.shape[0]
                                - int(np.ceil(-self.dy)),
                                :,
                            ]
                        else:
                            rectified_map = rectified_map[
                                int(np.floor(self.dy)) : map_img[
                                    "image"
                                ].data.shape[0],
                                :,
                            ]
                        # Generate dataset
                        rectified_map_dataset = create_dem(
                            rectified_map,
                            transform=map_img.georef_transform.data,
                            img_crs=map_img.crs,
                            no_data=-32768,
                        )

                        self.reproject_path[map_name] = os.path.join(
                            self.stats_dir, map_name + "_support_map_rectif.tif"
                        )

                        save_dem(
                            rectified_map_dataset, self.reproject_path[map_name]
                        )
                else:
                    self.reproject_path[map_name] = map_path


class FusionPartition(Partition):
    """
    FusionPartition
    TODO : comment
    TODO : clean pylint protected access
    """

    # pylint: disable=protected-access

    def __init__(self, partitions, output_dir, geo_ref=True):
        """
        TODO Merge the layers to generate the layers fusion
        :param partitions: list d objet Partition
        :return: TODO
        """

        # Sanity check
        if len(partitions) == 1:
            logging.error(
                "There must be at least 2 partitions to be merged together"
            )
            raise NotEnoughDataToPartitionError

        self.partitions = partitions

        self.dict_fusion = {
            "ref": np.all(
                [p.reproject_path["ref"] is not None for p in self.partitions]
            ),
            "dsm": np.all(
                [p.reproject_path["dsm"] is not None for p in self.partitions]
            ),
        }
        if ~(self.dict_fusion["ref"] + self.dict_fusion["dsm"]):
            logging.error(
                "For the partition to be merged, "
                "there must be at least one support (ref or dsm) "
                "provided by every partition"
            )
            raise NotEnoughDataToPartitionError

        super().__init__(
            "fusion_layer",
            "classification_layers",
            coreg_dsm=partitions[0].coreg_path["dsm"],
            coreg_ref=partitions[0].coreg_path["ref"],
            output_dir=output_dir,
            geo_ref=geo_ref,
        )

    def _create_partition_sets(self, **_):
        """
        Creates partitions sets
        :param kwargs:
        :return: None
        """
        self._fill_sets_attributes()
        self._set_labelled_map()

    def _set_labelled_map(self):
        """
        Sets labelled map
        :return: None
        """
        for (
            df_k,
            df_v,
        ) in (
            self.dict_fusion.items()
        ):  # df_k is 'ref' or 'dsm' and df_v is 'True' or 'False'
            if df_v:
                map_fusion = np.ones(self._coreg_shape) * -32768.0
                for label_idx, label_name in enumerate(self.sets_labels):
                    map_fusion[
                        self._sets_indexes[df_k][label_idx]
                    ] = self._classes[label_name]

                self.map_path[df_k] = os.path.join(
                    self.stats_dir, "{}_fusion_layer.tif".format(df_k)
                )

                save_dem(
                    self.partitions[0].coreg_path["ref"],
                    self.map_path[df_k],
                    new_array=map_fusion,
                    no_data=-32768,
                )

    def _fill_sets_attributes(self):
        """
        Fills sets attributes
        :return: None
        """
        all_combi_labels, self._classes = self._create_merged_classes(
            self.partitions
        )

        # create sets names and labels from classes
        self._fill_sets_names_labels()

        # create colors for every label
        self._sets_colors = (
            np.multiply(get_color(len(self.sets_names)), 255) / 255
        )

        # find out indexes for every label
        dict_partitions = {p.name: p for p in self.partitions}
        for (
            df_k,
            df_v,
        ) in (
            self.dict_fusion.items()
        ):  # df_k is 'ref' or 'dsm' and df_v is 'True' or 'False'
            if df_v:
                self._sets_indexes[df_k] = []
                for combi in all_combi_labels:
                    # following list will contain indexes for couple
                    # (partition layer, label index) for this merged label
                    all_indexes = []
                    for elm in combi:
                        layer_name = elm[0]
                        label_idx = elm[1]
                        all_indexes.append(
                            dict_partitions[layer_name]._sets_indexes[df_k][
                                label_idx
                            ]
                        )
                    for indexes2d in all_indexes:
                        np.ravel_multi_index(indexes2d, self._coreg_shape)
                    # ravel indexes so we can merge them
                    all_indexes = [
                        np.ravel_multi_index(indexes2d, self._coreg_shape)
                        for indexes2d in all_indexes
                    ]

                    # merge indexes and unravel them
                    merged_indexes = reduce(np.intersect1d, all_indexes)
                    self._sets_indexes[df_k].append(
                        np.unravel_index(merged_indexes, self._coreg_shape)
                    )

    @staticmethod
    def _create_merged_classes(partitions):
        """
        Generate the 'classes' dictionary for merged layers
        :param classes_to_fusion: list of classes to merge
        :return: TODO list of combinations of labels, new classes
        """

        classes_to_merge = []
        for partition in partitions:
            classes_to_merge.append(
                [
                    (
                        partition.name,
                        label_idx,
                        partition._sets_names[label_idx],
                    )
                    for label_idx in range(len(partition._sets_names))
                ]
            )

        # calcul toutes les combinaisons (developpement des labels entre eux)
        all_combi_labels = list(itertools.product(*classes_to_merge))

        new_label_value = 1
        new_classes = collections.OrderedDict()
        for combi in all_combi_labels:
            # creer le new label dans le dictionnaire new_classes
            new_label_name = "_&_".join(
                [
                    "_".join([name, str(label).split(":", maxsplit=1)[0]])
                    for name, value, label in combi
                ]
            )

            new_classes[new_label_name] = new_label_value
            new_label_value += 1

        return all_combi_labels, new_classes


def get_color(nb_color=10):
    """
    Function to get matplotlib color possibilities.
    :param nb_color: number of colors
    :type nb_color: int
    """

    if 10 < nb_color < 21:
        if matplotlib.__version__ >= "2.0.1":
            # According to matplotlib documentation the Vega colormaps are
            # deprecated since the 2.0.1 and disabled since 2.2.0
            x = mpl_pyplot.cm.get_cmap("tab20")
        else:
            x = mpl_pyplot.cm.get_cmap("Vega20")
    if nb_color < 11:
        if matplotlib.__version__ >= "2.0.1":
            x = mpl_pyplot.cm.get_cmap("tab10")
        else:
            x = mpl_pyplot.cm.get_cmap("Vega10")
    if nb_color > 20:
        clr = mpl_pyplot.cm.get_cmap("gist_earth")
        return np.array(
            [clr(c / float(nb_color))[0:3] for c in np.arange(nb_color)]
        )
    # else:
    return np.array(x.colors[0:nb_color])


def create_fusion(
    sets_masks: Dict, all_combi_labels, classes_fusion, layers_obj
):
    """
    TODO: create all maps fusion
    :param sets_masks: dict per layer (example 'slope', 'carte_occupation', ...)
    contains a tuple list each,
    where each tuple contains ('label_name', A3DGeoRaster_mask)
    :type sets_masks: Dict
    :param all_combi_labels:
    :type all_combi_labels: List
    :param classes_fusion:
    :type classes_fusion: List
    :param layers_obj: example layer to get size and georef
    :type layers_obj:
    layers_obj:
    :return:
    """
    # create map which fusion all classes combinaisons
    map_fusion = np.ones(layers_obj.r.shape) * -32768.0
    sets_fusion = []
    sets_colors = np.multiply(get_color(len(all_combi_labels)), 255)
    # get masks associated with tuples
    for combi in all_combi_labels:
        mask_fusion = np.ones(layers_obj.r.shape)
        for elm_combi in combi:
            layer_name = elm_combi[0]
            label_name = elm_combi[1]
            # concatenate masks of different labels
            # from tuple/combinaison in mask_fusion
            mask_label = np.zeros(layers_obj.r.shape)
            mask_label[sets_masks[layer_name][label_name]] = 1
            # TODO change sets_masks[layer_name]['sets_def'][label_name]
            # dict is not the same anymore => a list now
            mask_fusion = mask_fusion * mask_label

        # get new label associated in new_classes dict
        new_label_name = "&".join(["@".join(elm_combi) for elm_combi in combi])
        new_label_value = classes_fusion[new_label_name]
        map_fusion[np.where(mask_fusion)] = new_label_value
        # save mask_fusion
        sets_fusion.append((new_label_name, np.where(mask_fusion)))

    # save map fusion
    map_return = create_dem(
        map_fusion,
        transform=layers_obj.georef_transform.data,
        img_crs=layers_obj.crs,
        no_data=-32768,
    )

    return map_return, sets_fusion, sets_colors / 255.0
