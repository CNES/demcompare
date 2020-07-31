#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2019 Centre National d'Etudes Spatiales (CNES)

"""
Mainly contains the Partition class. A partition defines a way to partition the DEMs alti differences.
TODO add comment Fusion_partition
"""

import os
import collections
import numpy as np
import logging
from functools import reduce
from osgeo import gdal
import itertools

from .output_tree_design import get_out_dir
from .a3d_georaster import A3DGeoRaster, write_geotiff
from .a3d_raster_basic import A3DRasterBasic, write_tiff

class NotEnoughDataToPartitionError(Exception):
    pass


class Partition(object):
    class LackOfPartitionDataError(Exception):
        def __init__(self):
            logging.error('At least one partition support must be provided '
                          '(shall it be linked to the reference DSM or the slave one). '
                          'Use the \'ref\' and/or \'dsm\' keys respectively.')

    # Only kind of partition supported
    # TODO see if two classes will not be better
    type = ["to_be_classification_layers", "classification_layers"]

    # default no data value
    nodata = -32768.0

    ####### initialization #######
    def __init__(self, name, partition_kind, coreg_dsm, coreg_ref, outputDir, geo_ref=True, **cfg_layer):

        # Sanity check
        if partition_kind in self.type:
            self._type_layer = partition_kind
        else:
            logging.error('Unsupported partition kind {}. Try one of the following {}'.format(partition_kind, self.type))
            raise KeyError

        # Get partition name
        self._name = name

        # Create output dir (where to store partition results & data)
        self._output_dir = outputDir
        self.create_output_dir()

        # Store coreg path (TODO why?)
        self.coreg_path = {'ref': None, 'dsm': None}
        self._coreg_shape = coreg_ref.r.shape
        self.coreg_path['dsm'] = coreg_dsm
        self.coreg_path['ref'] = coreg_ref

        # Init input data path
        self.ref_path = ''
        self.dsm_path = ''

        # Init labelled map data
        self.reproject_path = {'ref': None, 'dsm': None}
        self.map_path = {'ref': None, 'dsm': None}

        # Init sets attributes
        self._sets_indexes = {'ref': None, 'dsm': None}
        self._sets_names = None
        self._sets_labels = None
        self._sets_masks = None

        # Georef set
        self.geo_ref = geo_ref

        # Create partition (labelled map with associated sets)
        self._create_patition_sets(**cfg_layer)



        logging.info('Partition created as: {}'.format(self))

    def _create_patition_sets(self, **cfg_layer):
        """

        :param cfg_layer:
        :return:
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
        self._sets_masks = [~ (np.isnan(self.coreg_path['dsm'].r) * np.isnan(self.coreg_path['ref'].r))]
        self._sets_colors = None

    ####### getters and setters #######
    @property
    def out_dir(self):
        return self._output_dir

    @property
    def stats_dir(self):
        return os.path.join(self.out_dir, get_out_dir('stats_dir'), self._name)

    @property
    def histograms_dir(self):
        return os.path.join(self.out_dir, get_out_dir('histograms_dir'), self._name)

    @property
    def plots_dir(self):
        return os.path.join(self.out_dir, get_out_dir('snapshots_dir'), self._name)

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
        return self._sets_indexes['ref']

    @property
    def sets_indexes_dsm(self):
        return self._sets_indexes['dsm']

    @property
    def stats_results(self):
        stats_results = {}
        #### mode standard ####
        stats_results['standard'] = {'Ref_support': None, 'DSM_support': None}
        if self.ref_path or self.map_path['ref']:
            stats_results['standard']['Ref_support'] = {'nodata': self.nodata, 'path': self.map_path['ref']}
        if self.dsm_path or self.map_path['dsm']:
            stats_results['standard']['DSM_support'] = {'nodata': self.nodata, 'path': self.map_path['dsm']}
        if (self.ref_path or self.map_path['ref']) and (self.dsm_path or self.map_path['dsm']):
            ##### mode coherent ######
            stats_results['coherent-classification'] = {'Ref_support': None, 'DSM_support': None}
            # TODO replire le coherent avec les map correspondante
            ##### mode incoherent ######
            stats_results['incoherent-classification'] = {'Ref_support': None, 'DSM_support': None}
            # TODO replire le coherent avec les map correspondante
        return stats_results

    @property
    def sets_masks(self):
        if self._sets_masks is None:
            all_masks = []
            ref_masks = []
            dsm_masks = []
            if self.sets_indexes_ref:
                for label_idx in range(len(self.sets_labels)):
                    ref_masks.append(np.ones(self.coreg_shape) * False)
                    ref_masks[label_idx][self.sets_indexes_ref[label_idx]] = True
                all_masks.append(ref_masks)
            if self.sets_indexes_dsm:
                for label_idx in range(len(self.sets_labels)):
                    dsm_masks.append(np.ones(self.coreg_shape) * False)
                    dsm_masks[label_idx][self.sets_indexes_dsm[label_idx]] = True
                all_masks.append(dsm_masks)

            self._sets_masks = all_masks

        return self._sets_masks

    def __repr__(self):
        return '\n'.join(["",
                          "----",
                          "| Partition `{}` of type : {}".format(self.name, self.type_layer),
                          "| - path to REF input: ",
                          "|\t{}".format(self.ref_path),
                          "|   whose labeled & coregistered version is ",
                          "|\t{}".format(self.map_path['ref']),
                          "| - path to DSM input:",
                          "|\t{}".format(self.dsm_path),
                          "|   whose labeled & coregistered version is ",
                          "|\t{}".format(self.map_path['dsm']),
                          "----"])

    def generate_classes(self, ranges):
        # change the intervals into a list to make 'classes' generic
        classes = collections.OrderedDict()
        for idx in range(0, len(ranges)):
            if idx == len(ranges) - 1:
                if self.name == 'slope':
                    key = "[{}%;inf[".format(ranges[idx])
                else:
                    key = "[{};inf[".format(ranges[idx])
            else:
                if self.name == 'slope':
                    key = "[{}%;{}%[".format(ranges[idx], ranges[idx + 1])
                else:
                    key = "[{};{}[".format(ranges[idx], ranges[idx + 1])
            classes[key] = ranges[idx]

        return classes

    def create_output_dir(self):
        """
        Create folder stats results
        :param name_layer: layer name
        :return:
        """
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.histograms_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def create_slope(self, coreg_dsm, coreg_ref):
        """
        Create slope if not exist
        :param coreg_dsm: A3DDEMRaster, input coregistered DSM
        :param coreg_ref: A3DDEMRaster, input coregistered REF
        :param name_layer: layer name
        :return:
        """
        # Compute ref slope
        self.ref_path = os.path.join(self.stats_dir, 'Ref_support.tif')
        slope_ref, aspect_ref = coreg_ref.get_slope_and_aspect(degree=False)
        if self.geo_ref:
            slope_ref_georaster = A3DGeoRaster.from_raster(slope_ref,
                                                           coreg_ref.trans,
                                                           "{}".format(coreg_ref.srs.ExportToProj4()),
                                                           nodata=self.nodata)
            slope_ref_georaster.save_geotiff(self.ref_path)
        else:
            slope_ref_georaster = A3DRasterBasic(slope_ref, nodata=self.nodata)
            slope_ref_georaster.save_tiff(self.ref_path)

        # Compute dsm slope
        self.dsm_path = os.path.join(self.stats_dir, 'DSM_support.tif')
        slope_dsm, aspect_dsm = coreg_dsm.get_slope_and_aspect(degree=False)
        if self.geo_ref:
            slope_dsm_georaster = A3DGeoRaster.from_raster(slope_dsm,
                                                           coreg_dsm.trans,
                                                           "{}".format(coreg_dsm.srs.ExportToProj4()),
                                                           nodata=self.nodata)
            slope_dsm_georaster.save_geotiff(self.dsm_path)
        else:
            slope_dsm_georaster = A3DRasterBasic(slope_dsm, nodata=self.nodata)
            slope_dsm_georaster.save_tiff(self.dsm_path)

    def create_map(self, slope_img, type_slope):
        """
        Create the map for each slope (l'intervalle des valeurs est transforme en 1 valeur (la min de l'intervalle))
        :param slope_img: slope image path
        :param type_slope: type of slope : 'ref' or 'dsm'
        :return:
        """
        # use radiometric ranges to classify
        slope = A3DGeoRaster(slope_img)
        rad_range = list(self.classes.values())
        if self.geo_ref:
            map_img = A3DGeoRaster.from_raster(np.ones(slope.r.shape) * self.nodata,
                                               slope.trans, "{}".format(slope.srs.ExportToProj4()), nodata=self.nodata)
        else:
            map_img = A3DRasterBasic(np.ones(slope.r.shape) * self.nodata)
        for idx in range(0, len(rad_range)):
            if idx == len(rad_range) - 1:
                map_img.r[np.where((~np.isnan(slope.r))*(slope.r >= rad_range[idx]))] = rad_range[idx]
            else:
                map_img.r[np.where((~np.isnan(slope.r))*(slope.r >= rad_range[idx]) & (slope.r < rad_range[idx + 1]))] = rad_range[idx]

        self.map_path[type_slope] = os.path.join(self.stats_dir, type_slope + '_support_map.tif')
        if self.geo_ref:
            map_img.save_geotiff(self.map_path[type_slope])
        else:
            map_img.save_tiff(self.map_path[type_slope])

    def _create_set_indices(self):
        """
        Returns a list of numpy.where, by class. Each element defines a set. The sets partition / classify the image.
        Each numpy.where contains the coordinates of the sets of the class.
        Create list of coordinates arrays :
            -> self.sets_indices = [(label_name, np.where(...)), ... label_name, np.where(...))] ,

        :param classes: ordered dict of labels and associated values
        :return:
        """
        dsm_supports = ['ref', 'dsm']
        sets_indices = {support: None for support in dsm_supports }
        for support in dsm_supports:
            sets_indices[support] = []
            bool_set_ind = False
            if self.geo_ref:
                if self.reproject_path[support]:
                    img_to_classify = A3DGeoRaster(self.reproject_path[support]).r
                    bool_set_ind = True
            else:
                if self.map_path[support]:
                    img_to_classify = A3DRasterBasic.from_path(str(self.map_path[support])).r
                bool_set_ind = True
            if bool_set_ind:
                # calculate sets_indices of partition
                for class_name, class_value in self.classes.items():
                    if isinstance(class_value, list):
                        if len(class_value) == 1:
                            # transform it to value
                            class_value = class_value[0]
                    if isinstance(class_value, list):
                        elm = (class_name, np.where(np.logical_or(*[np.equal(img_to_classify, label_i)
                                                                     for label_i in class_value])))
                    else:
                        elm = (class_name, np.where(img_to_classify == class_value))
                    sets_indices[support].append(elm)

        return sets_indices

    def _fill_sets_names_labels(self):
        """

        :return:
        """
        # fill sets_labels & sets_names
        if self.name == 'slope':
            self._sets_names = list(self.classes.keys())
            # Slope labels are historically customized
            self._sets_labels = [r'$\nabla$ > {}%'.format(self.classes[set_name])
                                 if set_name.endswith('inf[') else r'$\nabla \in$ {}'.format(set_name)
                                 for set_name in self._sets_names]
        else:
            self._sets_labels = list(self.classes.keys())
            self._sets_names = ['{}:{}'.format(key, value) for key, value in self.classes.items()]
            self._sets_names = [name.replace(',', ';') for name in self._sets_names]

    def _fill_sets_attributes(self):
        """

        :return:
        """
        self._fill_sets_names_labels()

        # fill sets_colors
        self._sets_colors = np.multiply(getColor(len(self.sets_names)), 255) / 255

        # fill sets_indexes
        tuples_of_labels_and_indexes = self._create_set_indices()
        if tuples_of_labels_and_indexes['ref']:
            self._sets_indexes['ref'] = [item[1] for item in tuples_of_labels_and_indexes['ref']]
        if tuples_of_labels_and_indexes['dsm']:
            self._sets_indexes['dsm'] = [item[1] for item in tuples_of_labels_and_indexes['dsm']]

    def _create_labelled_map(self, **cfg_layer):

        # Store classes (TODO why?)
        self._classes = {}
        if 'classes' in cfg_layer:
            self._classes = collections.OrderedDict(cfg_layer['classes'])
        elif 'ranges' in cfg_layer:
            # transform 'ranges' to 'classes'
            self._classes = self.generate_classes(cfg_layer['ranges'])
        else:
            logging.error('Neither classes nor ranges where given as input sets to partition the stats')
            raise KeyError

        # Store path to initial layer
        if 'ref' in cfg_layer:
            self.ref_path = cfg_layer['ref']
        if 'dsm' in cfg_layer:
            self.dsm_path = cfg_layer['dsm']
        if (not 'ref' in cfg_layer) and (not 'dsm' in cfg_layer):
            raise self.LackOfPartitionDataError
        if (not self.ref_path) and (not self.dsm_path):
            if self.type_layer == "classification_layers":
                raise self.LackOfPartitionDataError
            else:
                if self.name != 'slope':
                    raise self.LackOfPartitionDataError
                else:
                    # create slope : ref and dsm
                    self.create_slope(self.coreg_path['dsm'], self.coreg_path['ref'])

        # Create the layer map
        if self.type_layer == "to_be_classification_layers":
            # if the partition is not yet a labelled map, then make it so
            if self.ref_path:
                self.create_map(self.ref_path, 'ref')
            if self.dsm_path:
                self.create_map(self.dsm_path, 'dsm')
        elif self.type_layer == "classification_layers":
            if 'ref' in cfg_layer:
                self.map_path['ref'] = cfg_layer['ref']
            if 'dsm' in cfg_layer:
                self.map_path['dsm'] = cfg_layer['dsm']

        # Reproj the layer map
        self.rectify_map()

    def rectify_map(self):
        """
        Reproject  the layer maps on top of coreg dsm and coreg ref (which are coregistered together)

        :return:
        """
        for map_name, map_path in self.map_path.items():
            if map_path:
                if self.geo_ref:
                    map_img = A3DGeoRaster(map_path)
                    rectified_map = map_img.reproject(self.coreg_path[map_name].srs, int(self.coreg_path[map_name].nx),
                                                      int(self.coreg_path[map_name].ny), self.coreg_path[map_name].footprint[0],
                                                      self.coreg_path[map_name].footprint[3],
                                                      self.coreg_path[map_name].xres, self.coreg_path[map_name].yres,
                                                      nodata=map_img.nodata, interp_type=gdal.GRA_NearestNeighbour)

                    self.reproject_path[map_name] = os.path.join(self.stats_dir,
                                                                 map_name + '_support_map_rectif.tif')
                    rectified_map.save_geotiff(self.reproject_path[map_name])
                else:
                    self.reproject_path[map_name] = map_path


class Fusion_partition(Partition):

    def __init__(self, partitions, outputDir, geo_ref=True):
        """
        TODO Merge the layers to generate the layers fusion
        :param partitions: list d objet Partition
        :return: TODO
        """

        # Sanity check
        if len(partitions) == 1:
            logging.error('There must be at least 2 partitions to be merged together')
            raise NotEnoughDataToPartitionError

        self.partitions = partitions

        self.dict_fusion = {'ref': np.all([p.reproject_path['ref'] is not None for p in self.partitions]),
                       'dsm': np.all([p.reproject_path['dsm'] is not None for p in self.partitions])}
        if ~(self.dict_fusion['ref'] + self.dict_fusion['dsm']) :
            logging.error('For the partition to be merged, there must be at least one support (ref or dsm) '
                          'provided by every partition')
            raise NotEnoughDataToPartitionError

        super(Fusion_partition, self).__init__('fusion_layer',
                                               'classification_layers',
                                               coreg_dsm=partitions[0].coreg_path['dsm'],
                                               coreg_ref=partitions[0].coreg_path['ref'],
                                               outputDir=outputDir, geo_ref=geo_ref)

    def _create_patition_sets(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self._fill_sets_attributes()
        self._set_labelled_map()

    def _set_labelled_map(self):
        for df_k, df_v in self.dict_fusion.items():  #df_k is 'ref' or 'dsm' and df_v is 'True' or 'False'
            if df_v:
                map_fusion = np.ones(self._coreg_shape) * -32768.0
                for label_idx, label_name in enumerate(self.sets_labels):
                    map_fusion[self._sets_indexes[df_k][label_idx]] = self._classes[label_name]

                self.map_path[df_k] = os.path.join(self.stats_dir, '{}_fusion_layer.tif'.format(df_k))
                if self.geo_ref:
                    write_geotiff(self.map_path[df_k], map_fusion,
                                  self.partitions[0].coreg_path['ref'].trans,
                                  wkt=self.partitions[0].coreg_path['ref'].srs.ExportToWkt(),
                                  nodata=-32768.0)
                else:
                    write_tiff(self.map_path[df_k], map_fusion,
                                  nodata=-32768.0)

    def _fill_sets_attributes(self):
        all_combi_labels, self._classes = self._create_merged_classes(self.partitions)

        # create sets names and labels from classes
        self._fill_sets_names_labels()

        # create colors for every label
        self._sets_colors = np.multiply(getColor(len(self.sets_names)), 255) / 255

        # find out indexes for every label
        dict_partitions = {p.name: p for p in self.partitions}
        for df_k, df_v in self.dict_fusion.items():  #df_k is 'ref' or 'dsm' and df_v is 'True' or 'False'
            if df_v:
                self._sets_indexes[df_k] = []
                for combi in all_combi_labels:
                    # following list will contain indexes for couple (partition layer, label index) for this merged label
                    all_indexes = []
                    for elm in combi:
                        layer_name = elm[0]
                        label_idx = elm[1]
                        all_indexes.append(dict_partitions[layer_name]._sets_indexes[df_k][label_idx])

                    # ravel indexes so we can merge them
                    all_indexes = [np.ravel_multi_index(indexes2D, self._coreg_shape) for indexes2D in all_indexes]

                    # merge indexes and unravel them
                    merged_indexes = reduce(np.intersect1d, all_indexes)
                    self._sets_indexes[df_k].append(np.unravel_index(merged_indexes, self._coreg_shape))


    def _create_merged_classes(self, partitions):
        """
        Generate the 'classes' dictionary for merged layers
        :param classes_to_fusion: list of classes to merge
        :return: TODO list of combinations of labels, new classes
        """

        classes_to_merge = []
        for partition in partitions:
            classes_to_merge.append(
                [(partition.name, label_idx, partition._sets_names[label_idx]) for label_idx in range(len(partition._sets_names))])

        # calcul toutes les combinaisons (developpement des labels entre eux)
        all_combi_labels = list(itertools.product(*classes_to_merge))

        new_label_value = 1
        new_classes = collections.OrderedDict()
        for combi in all_combi_labels:
            # creer le new label dans le dictionnaire new_classes
            new_label_name = '_&_'.join(['_'.join([name, str(label).split(':')[0]]) for name, value, label in combi])

            new_classes[new_label_name] = new_label_value
            new_label_value += 1

        return all_combi_labels, new_classes


def getColor(nb_color=10):
    import matplotlib
    import matplotlib.pyplot as P
    if 10 < nb_color < 21:
        if matplotlib.__version__ >= '2.0.1':
            # According to matplotlib documentation the Vega colormaps are deprecated since the 2.0.1 and
            # disabled since 2.2.0
            x = P.cm.get_cmap('tab20')
        else:
            x = P.cm.get_cmap('Vega20')
    if nb_color < 11:
        if matplotlib.__version__ >= '2.0.1':
            x = P.cm.get_cmap('tab10')
        else:
            x = P.cm.get_cmap('Vega10')
    if nb_color > 20:
        clr = P.cm.get_cmap('gist_earth')
        return np.array([clr(c/float(nb_color))[0:3] for c in np.arange(nb_color)])
    else:
        return np.array(x.colors[0:nb_color])


def create_fusion(sets_masks, all_combi_labels, classes_fusion, layers_obj):
    """
    TODO create la fusion de toute les maps
    :param sets_masks: dict par layer (exemple 'slope', 'carte_occupation', ...) contentant chacun une liste de tuple,
                        dont chaque tuple contient ('nom_label', A3DGeoRaster_mask)
    layers_obj: une layer d'exemple pour recuperer la taille et le georef
    :return:
    """
    # create map qui fusionne toutes les combinaisons de classes
    map_fusion = np.ones(layers_obj.r.shape) * -32768.0
    sets_fusion = []
    sets_colors = np.multiply(getColor(len(all_combi_labels)), 255)
    # recupere les masques associées aux tuples
    for combi in all_combi_labels:
        mask_fusion = np.ones(layers_obj.r.shape)
        for elm_combi in combi:
            layer_name = elm_combi[0]
            label_name = elm_combi[1]
            # concatene les masques des differentes labels du tuple/combi dans mask_fusion
            mask_label = np.zeros(layers_obj.r.shape)
            mask_label[sets_masks[layer_name][label_name]] = 1              # TODO change sets_masks[layer_name]['sets_def'][label_name] le dictionnaire n'est plus le meme => une liste mtn
            mask_fusion = mask_fusion * mask_label

        # recupere le new label associé dans ls dictionnaire new_classes
        new_label_name = '&'.join(['@'.join(elm_combi) for elm_combi in combi])
        new_label_value = classes_fusion[new_label_name]
        map_fusion[np.where(mask_fusion)] = new_label_value
        # save mask_fusion
        sets_fusion.append((new_label_name, np.where(mask_fusion)))

    # save map fusionne
    map = A3DGeoRaster.from_raster(map_fusion,layers_obj.trans,
                                   "{}".format(layers_obj.srs.ExportToProj4()), nodata=-32768)

    return map, sets_fusion, sets_colors / 255.
