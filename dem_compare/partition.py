#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2019 Centre National d'Etudes Spatiales (CNES)

"""
TODO
partition class
"""

import os
import collections
import numpy as np

from .output_tree_design import get_out_dir
from .a3d_georaster import A3DGeoRaster


class Partition:

    ####### attributs #######
    name = ''
    #type
    type = ["to_be_classification_layers", "classification_layers"]
    type_layer = None  # slope "to_be_classification_layers" or map "classification_layers"

    # in paths
    ref_path = ''
    dsm_path = ''

    # rectified paths
    ref_reproject_path = ''
    dsm_reproject_path = ''

    # map paths
    ref_map_path = ''
    dsm_map_path = ''

    nodata = -32768.0

    # classes (dictionary contain labels and names)
    classes = {}

    # sets
    sets_masks = None
    sets_colors = None

    sets_names = None
    sets_labels = None

    ####### initialization #######
    def __init__(self, name_layer, type_layer, coreg_dsm, coreg_ref, outputDir, **cfg_layer):
        self.name = name_layer

        if type_layer in type:
            self.layer_type = type_layer
        else:
            # TODO execption
            raise('type_layer est faux! {}'.format(type_layer))

        if 'ref' in cfg_layer:
            self.ref_path = cfg_layer['ref']
        if 'dsm' in cfg_layer:
            self.dsm_path = cfg_layer['dsm']
        if (not 'ref' in cfg_layer) and (not 'dsm' in cfg_layer) and (self.layer_type == "to_be_classification_layers"):
            # TODO Create les 2 slopes
            # create slope : ref and dsm
            self.create_slope(coreg_dsm, coreg_ref, outputDir, self.name)
        else:
            if (not 'ref' in cfg_layer) and (not 'dsm' in cfg_layer):
                raise("PBM ni de REF ni de DSM")

        if 'classes' in cfg_layer:
            self.classes = cfg_layer['classes']
        elif 'ranges' in cfg_layer:
            # transform 'ranges' to 'classes'
            self.classes = self.generate_classes(cfg_layer['ranges'])
        else:
            raise("PBM ni de ranges ni de classes!!!")

        # slope transform to map
        if self.layer_type == "to_be_classification_layers":
            if self.ref_path:
                # slope transform
                self.ref_map_path = self.create_map(self.ref_path, 'ref', outputDir)
            if self.dsm_path:
                self.dsm_map_path = self.create_map(self.dsm_path, 'dsm', outputDir)

        elif self.layer_type == "classification_layers":
            if 'ref' in cfg_layer:
                self.ref_map_path = cfg_layer['ref']
            if 'dsm' in cfg_layer:
                self.dsm_map_path = cfg_layer['dsm']

        # TODO poursuivre l'init de partition

    ####### getters and setters #######
    #def get_sets_masks():
    #def get_sets_colors():
    #def get_sets_names():
    #def get_sets_labels():


    ####### others #######
    def generate_classes(self, ranges):
        # change the intervals into a list to make 'classes' generic
        classes = collections.OrderedDict()
        for idx in range(0, len(ranges)):
            if idx == len(ranges) - 1:
                key = "[{};inf]".format(ranges[idx])
            else:
                key = "[{};{}[".format(ranges[idx], ranges[idx + 1])
            classes[key] = ranges[idx]

        self.classes = classes

    def create_stats_results(self, outputDir, name_layer):
        """
        Create folder stats results
        :param outputDir: output directory
        :param name_layer: layer name
        :return:
        """
        os.makedirs(os.path.join(outputDir, get_out_dir('stats_dir'), name_layer), exist_ok=True)

    def create_slope(self, coreg_dsm, coreg_ref, outputDir, name_layer):
        """
        Create slope if not exist
        :param coreg_dsm: A3DDEMRaster, input coregistered DSM
        :param coreg_ref: A3DDEMRaster, input coregistered REF
        :param outputDir: output directory
        :param name_layer: layer name
        :return:
        """
        # Compute ref slope
        self.ref_path = os.path.join(outputDir, get_out_dir('stats_dir'), name_layer, 'Ref_support.tif')
        slope_ref, aspect_ref = coreg_ref.get_slope_and_aspect(degree=False)
        slope_ref_georaster = A3DGeoRaster.from_raster(slope_ref,
                                                       coreg_ref.trans,
                                                       "{}".format(coreg_ref.srs.ExportToProj4()),
                                                       nodata=self.nodata)
        slope_ref_georaster.save_geotiff(self.ref_path)

        # Compute dsm slope
        self.dsm_path = os.path.join(outputDir, get_out_dir('stats_dir'), name_layer, 'DSM_support.tif')
        slope_dsm, aspect_dsm = coreg_dsm.get_slope_and_aspect(degree=False)
        slope_dsm_georaster = A3DGeoRaster.from_raster(slope_dsm,
                                                       coreg_dsm.trans,
                                                       "{}".format(coreg_dsm.srs.ExportToProj4()),
                                                       nodata=self.nodata)
        slope_dsm_georaster.save_geotiff(self.dsm_path)

        return slope_ref_georaster, slope_dsm_georaster

    def create_map(self, slope_img, outputDir, type_slope):
        """
        Create the map for each slope (l'intervalle des valeurs est transforme en 1 valeur (la min de l'intervalle))
        :param slope_img: slope image path
        :param outputDir: output directory
        :param type_slope: type of slope : 'ref' or 'dsm'
        :return:
        """
        # use radiometric ranges to classify
        slope = A3DGeoRaster(slope_img)
        rad_range = self.classes.values()
        map_img = A3DGeoRaster.from_raster(np.ones(slope.r.shape) * self.nodata,
                                           slope.trans, "{}".format(slope.srs.ExportToProj4()), nodata=self.nodata)
        for idx in range(0, len(rad_range)):

            if idx == len(rad_range) - 1:
                map_img.r[np.where((slope.r >= rad_range[idx]))] = rad_range[idx]
            else:
                map_img.r[np.where((slope.r >= rad_range[idx]) & (slope.r < rad_range[idx + 1]))] = rad_range[idx]

        map_path = os.path.join(outputDir, get_out_dir('stats_dir'), self.name, type_slope + '_support_map.tif')
        map_img.save_geotiff(map_path)

        return map_path
