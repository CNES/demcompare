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
from osgeo import gdal
import itertools

from .output_tree_design import get_out_dir
from .a3d_georaster import A3DGeoRaster


class Partition:

    ####### initialization #######
    def __init__(self, name_layer, type_layer, coreg_dsm, coreg_ref, outputDir, **cfg_layer):

        ####### attributs objet #######
        # type
        self.type = ["to_be_classification_layers", "classification_layers"]
        self.type_layer = None

        # output directory path
        self.output_dir = ''

        # in paths
        self.ref_path = ''
        self.dsm_path = ''
        self.coreg_path = {'ref': None, 'dsm': None}

        # rectified paths
        self.reproject_path = {'ref': None, 'dsm': None}
        # map paths
        self.map_path = {'ref': None, 'dsm': None}

        self.nodata = -32768.0

        # classes (dictionary contain labels and names)
        self.classes = {}

        # sets
        self.sets_indices = {'ref': None, 'dsm': None}
        self.sets_colors = None

        self.sets_names = None
        self.sets_labels = None

        ########### init ############
        self.name = name_layer

        if type_layer in self.type:
            self.type_layer = type_layer
        else:
            raise('type_layer est faux! {}'.format(type_layer))

        self.output_dir = outputDir

        # create stats results folder
        self.create_stats_results()

        # get layer ref and/or dsm
        if 'ref' in cfg_layer:
            self.ref_path = cfg_layer['ref']
        if 'dsm' in cfg_layer:
            self.dsm_path = cfg_layer['dsm']

        if 'classes' in cfg_layer:
            self.classes = cfg_layer['classes']
        elif 'ranges' in cfg_layer:
            # transform 'ranges' to 'classes'
            self.generate_classes(cfg_layer['ranges'])
        else:
            raise("PBM ni de ranges ni de classes!!!")

        if (not cfg_layer['ref']) and (not cfg_layer['dsm']) and (self.type_layer == "to_be_classification_layers"):
            # create slope : ref and dsm
            self.create_slope(coreg_dsm, coreg_ref)
        elif (not 'ref' in cfg_layer) and (not 'dsm' in cfg_layer):
            raise("PBM ni de REF ni de DSM")

        if self.dsm_path:
          self.coreg_path['dsm'] = coreg_dsm
        if self.ref_path:
          self.coreg_path['ref'] = coreg_ref

        # slope transform to map
        if self.type_layer == "to_be_classification_layers":
            if self.ref_path:
                # slope transform
                self.create_map(self.ref_path, 'ref')
            if self.dsm_path:
                self.create_map(self.dsm_path, 'dsm')

        elif self.type_layer == "classification_layers":
            if 'ref' in cfg_layer:
                self.map_path['ref'] = cfg_layer['ref']
            if 'dsm' in cfg_layer:
                self.map_path['dsm'] = cfg_layer['dsm']

        print('OBJ Partition créé :', self.get_attrib())

    ####### getters and setters #######
    def get_name(self):
        return self.name

    def get_type_layer(self):
        return self.type_layer

    def get_sets_indices(self):
        return self.sets_indices

    def get_sets_indices(self, tlayer):
        # tlayer = 'ref' or 'dsm'
        return self.sets_indices[tlayer]

    def get_sets_colors(self):
        return self.sets_colors

    def get_sets_names(self):
        return self.sets_names

    def get_sets_labels(self):
        return self.sets_labels

    def get_attrib(self):
        print("self.name : {}\n, self.type_layer : {}\n, self.ref_path : {}\n, self.dsm_path : {}\n, "
              "self.reproject_path : {}\n, self.map_path : {}\n, self.classes : {}\n, self.coreg_path : {}\n,"
              "self.sets_indices : {}\n, self.sets_colors : {} \n".format(
              self.name, self.type_layer, self.ref_path, self.dsm_path,
              self.reproject_path, self.map_path, self.classes, self.coreg_path, self.sets_indices, self.sets_colors))

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

    def create_stats_results(self):
        """
        Create folder stats results
        :param name_layer: layer name
        :return:
        """
        os.makedirs(os.path.join(self.output_dir, get_out_dir('stats_dir'), self.name), exist_ok=True)

    def create_slope(self, coreg_dsm, coreg_ref):
        """
        Create slope if not exist
        :param coreg_dsm: A3DDEMRaster, input coregistered DSM
        :param coreg_ref: A3DDEMRaster, input coregistered REF
        :param name_layer: layer name
        :return:
        """
        # Compute ref slope
        self.ref_path = os.path.join(self.output_dir, get_out_dir('stats_dir'), self.name, 'Ref_support.tif')
        slope_ref, aspect_ref = coreg_ref.get_slope_and_aspect(degree=False)
        slope_ref_georaster = A3DGeoRaster.from_raster(slope_ref,
                                                       coreg_ref.trans,
                                                       "{}".format(coreg_ref.srs.ExportToProj4()),
                                                       nodata=self.nodata)
        slope_ref_georaster.save_geotiff(self.ref_path)

        # Compute dsm slope
        self.dsm_path = os.path.join(self.output_dir, get_out_dir('stats_dir'), self.name, 'DSM_support.tif')
        slope_dsm, aspect_dsm = coreg_dsm.get_slope_and_aspect(degree=False)
        slope_dsm_georaster = A3DGeoRaster.from_raster(slope_dsm,
                                                       coreg_dsm.trans,
                                                       "{}".format(coreg_dsm.srs.ExportToProj4()),
                                                       nodata=self.nodata)
        slope_dsm_georaster.save_geotiff(self.dsm_path)

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
        map_img = A3DGeoRaster.from_raster(np.ones(slope.r.shape) * self.nodata,
                                           slope.trans, "{}".format(slope.srs.ExportToProj4()), nodata=self.nodata)
        for idx in range(0, len(rad_range)):
            if idx == len(rad_range) - 1:
                map_img.r[np.where((slope.r >= rad_range[idx]))] = rad_range[idx]
            else:
                map_img.r[np.where((slope.r >= rad_range[idx]) & (slope.r < rad_range[idx + 1]))] = rad_range[idx]

        self.map_path[type_slope] = os.path.join(self.output_dir, get_out_dir('stats_dir'),
                                                 self.name, type_slope + '_support_map.tif')
        map_img.save_geotiff(self.map_path[type_slope])

    def rectify_map(self):
        """
        Rectification the maps for stats
        """
        #
        # Reproject image on top of coreg dsm and coreg ref (which are coregistered together)
        #
        for map_name,map_path in self.map_path.items():
            if map_path:
                map_img = A3DGeoRaster(map_path)
                rectified_map = map_img.reproject(self.coreg_path[map_name].srs, int(self.coreg_path[map_name].nx),
                                                  int(self.coreg_path[map_name].ny), self.coreg_path[map_name].footprint[0],
                                                  self.coreg_path[map_name].footprint[3],
                                                  self.coreg_path[map_name].xres, self.coreg_path[map_name].yres,
                                                  nodata=map_img.nodata, interp_type=gdal.GRA_NearestNeighbour)
                self.reproject_path[map_name] = os.path.join(self.output_dir, get_out_dir('stats_dir'), self.name,
                                                             map_name + '_support_map_rectif.tif')
                rectified_map.save_geotiff(self.reproject_path[map_name])

    def create_sets(self):
        """
        Returns a list of numpy.where, by class. Each element defines a set. The sets partition / classify the image.
        Each numpy.where contains the coordinates of the sets of the class.
        Create list of coordinates arrays and colors associated (RGB colors) :
            -> self.sets_indices = [(label_name, np.where(...)), ... label_name, np.where(...))] ,
            -> self.sets_colors = array([[0.12156863, 0.46666667, 0.70588235], ..., [0.7372549 , 0.74117647, 0.13333333]])
        """
        # calculate sets_colors of partition
        self.sets_colors = np.multiply(getColor(len(self.classes.values())), 255) / 255

        for l_type in ['ref', 'dsm']:
            if self.reproject_path[l_type]:
                img_to_classify = A3DGeoRaster(self.reproject_path[l_type])
                self.sets_indices[l_type] = []
                # calculate sets_indices of partition
                for class_name, class_value in self.classes.items():
                    if isinstance(class_value, list):
                        elm = (class_name, np.where(np.logical_or(*[np.equal(img_to_classify.r, label_i)
                                                                     for label_i in class_value])))
                    else:
                        elm = (class_name, np.where(img_to_classify.r == class_value))
                    self.sets_indices[l_type].append(elm)


############################### TODO a refac ###############################
    @classmethod
    def partition_fusion(cls, partitions):
        """
        TODO Merge the layers to generate the layers fusion
        :param partitions: list d objet Partition
        :return: TODO
        """

        def variables_activate(key, partitions):
            """
            TODO
            :param key:
            :param partitions:
            :return:
            """
            # key = 'ref' ou 'dsm'
            for partition in partitions:
                if not partition.reproject_path[key]:
                    return False
            return True

        # la fusion des layers (slope, map, ...) ne se fait que si toutes les layers sont renseignees (= reproject_path[ref/dsm] pas à None)
        all_layers_ref_flag = variables_activate('ref', partitions)
        all_layers_dsm_flag = variables_activate('dsm', partitions)
        print("all_layers_ref_flag, all_layers_dsm_flag = ", all_layers_ref_flag, all_layers_dsm_flag)

        dict_fusion = {'ref': all_layers_ref_flag, 'dsm': all_layers_dsm_flag}
        #support_name = {'ref': 'Ref_support', 'dsm': 'DSM_support'}
        #dict_stats_fusion = {'ref': None, 'dsm': None, 'reproject_ref': None, 'reproject_dsm': None,
        #                     'stats_results': {'Ref_support': None, 'DSM_support': None}}
        #dict_stats_fusion['stats_results'] = {'ref': None, 'dsm': None}
        classes_fusion = None
        all_combi_labels = None

        # create folder stats results fusion si layers_ref_flag ou layers_dsm_flag est à True
        # =====> On ne cree pas le dossier de sortie de la couche fusionné
        #if all_layers_ref_flag or all_layers_dsm_flag:
        #    create_stats_results(outputDir, 'fusion_layer')

        # Boucle sur [ref, dsm]
        #   Boucle sur chaque layer
        #       S'il y a plusieurs des layers données ou calculées
        #           ==> pour calculer les masks de chaque label
        #           ==> calculer toutes les combinaisons (developpement des labels entre eux mais pas les listes)
        #           ==> puis calculer les masks fusionnés (a associer avec les bons labels)
        #           ==> generer la nouvelle image classif (fusion) avec de nouveaux labels calculés arbitrairement et liés aux labels d entrees
        for df_k, df_v in dict_fusion.items():
            print("@@@@@@@@@@@@@@@@@@@")
            if df_v:
                # get les reproject_ref/dsm et faire une nouvelles map avec son dictionnaire associé
                clayers_to_fusion_path = [(parti.name, parti.reproject_path[df_k]) for parti in partitions]
                print("clayers_to_fusion_path = ", clayers_to_fusion_path)

                # lire les images clayers_to_fusion
                clayers_to_fusion = [(k, A3DGeoRaster(cltfp)) for k, cltfp in clayers_to_fusion_path]

                classes_to_fusion = []
                for partition in partitions:
                    classes_to_fusion.append([(partition.name, cl_classes_label) for cl_classes_label in partition.classes.keys()])

                if not (all_combi_labels and classes_fusion):
                    all_combi_labels, classes_fusion = create_new_classes(classes_to_fusion)
                print("all_combi_labels, classes_fusion = ", all_combi_labels, classes_fusion)

                # stop refac ici
                # create la layer fusionnee + les sets assossiés
                # TODO pour savoir qu est ce que l'on sait : on sait que l'on est dans ref OU dsm et on a partition
                # TODO          on veut sets_masks qui vaut : liste ordonnées des masks de tous les labels en FONCTION DE REF OU DSM
                # TODO get_sets_indices['ref'] ou 'dsm'
                sets_masks = [parti.get_sets_indices(df_k) for parti in partitions]                                         # TODO verififer que c'est ça!!!!
                print("sets_masks = ", sets_masks)
                map_fusion, sets_def_fusion, sets_colors_fusion = create_fusion(sets_masks, all_combi_labels, classes_fusion, clayers_to_fusion[0][1])
                sets_fusion = {df_k: {'fusion_layer': {'sets_def': dict(sets_def_fusion), 'sets_colors': sets_colors_fusion}}}

                # save map_fusion
                map_fusion_path = os.path.join(outputDir, get_out_dir('stats_dir'),
                                               'fusion_layer', '{}_fusion_layer.tif'.format(df_k))
                map_fusion.save_geotiff(map_fusion_path)
                # save dico de la layer
                dict_stats_fusion['classes'] = classes_fusion
                dict_stats_fusion[df_k] = map_fusion_path
                dict_stats_fusion[str('reproject_{}'.format(df_k))] = map_fusion_path
                dict_stats_fusion['stats_results'][support_name[df_k]] = {'nodata': -32768, 'path': map_fusion_path}

        # ajout des stats fussionees dans le dictionnaire
        clayers['fusion_layer'] = dict_stats_fusion

        # TODO return cls(avec les param de l'init sur la nouvelle partition + rajouter un param flag dans l'init 'rectified' pour savoir si la couche est déjà rectifié)

  # TODO sets_names, set_labels


#####################################################
######  fonction n'appartenant pas à la classe ######
#####################################################
def create_new_classes(classes_to_fusion):
    """
    Generate the 'classes' dictionary for merged layers
    :param classes_to_fusion: list of classes to merge
    :return: TODO list of combinations of labels, new classes
    """
    # calcul toutes les combinaisons (developpement des labels entre eux)
    all_combi_labels = list(itertools.product(*classes_to_fusion))

    new_label_value = 1
    new_classes = {}
    for combi in all_combi_labels:
        # creer le new label dans le dictionnaire new_classes
        new_label_name = '&'.join(['@'.join(elm_combi) for elm_combi in combi])
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
        raise NameError("Error : Too many colors requested")

    return np.array(x.colors)


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
        #dict_elm_to_fusion = {}
        mask_fusion = np.ones(layers_obj.r.shape)
        for elm_combi in combi:
            layer_name = elm_combi[0]
            label_name = elm_combi[1]
            # recupere le mask associé au label_name
            #dict_elm_to_fusion[layer_name] = {}
            #dict_elm_to_fusion[layer_name][label_name] = sets_masks[layer_name]['sets_def'][label_name]
            # concatene les masques des differentes labels du tuple/combi dans mask_fusion
            mask_label = np.zeros(layers_obj.r.shape)
            mask_label[sets_masks[layer_name]['sets_def'][label_name]] = 1              # TODO change sets_masks[layer_name]['sets_def'][label_name] le dictionnaire n'est plus le meme => une liste mtn
            mask_fusion = mask_fusion * mask_label

        # recupere le new label associé dans ls dictionnaire new_classes
        new_label_name = '&'.join(['@'.join(elm_combi) for elm_combi in combi])
        new_label_value = classes_fusion[new_label_name]
        map_fusion[np.where(mask_fusion)] = new_label_value
        # SAVE mask_fusion
        sets_fusion.append((new_label_name, np.where(mask_fusion)))

    # save map fusionne
    map = A3DGeoRaster.from_raster(map_fusion,layers_obj.trans,
                                   "{}".format(layers_obj.srs.ExportToProj4()), nodata=-32768)

    return map, sets_fusion, sets_colors / 255.