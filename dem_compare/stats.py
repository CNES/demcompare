#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
Stats module of dsm_compare offers routines for stats computation and plot viewing

"""

import os
import logging
import copy
import math
import json
import collections
import csv
import numpy as np
from osgeo import gdal
from scipy import exp
from scipy.optimize import curve_fit
from astropy import units as u


from .a3d_georaster import A3DGeoRaster
from .partition import Partition, Fusion_partition, getColor, NotEnoughDataToPartitionError
from .output_tree_design import get_out_dir, get_out_file_path


def gaus(x, a, x_zero, sigma):
    return a * exp(-(x - x_zero) ** 2 / (2 * sigma ** 2))


def roundUp(x, y):
    return int(math.ceil((x / float(y)))) * y


# DECREPATED (mais utilise dans dem_compare_extra)
def create_sets_slope(img_to_classify, sets_rad_range, tmpDir='.', output_descriptor=None):
    """
    Returns a list of boolean arrays. Each array defines a set. The sets partition / classify the image.
    A boolean array defines indices to kept for the associated set / class.
    :param img_to_classify: A3DGeoRaster
    :param sets_rad_range: list of values that defines the radiometric ranges of each set
    :param type: type of the stats calculate, 'slope' or 'classification'
    :param tmpDir: temporary directory to which store temporary data
    :param output_descriptor: dictionary with 'path' and 'nodata' keys for the output classified img (png format)
    :return: list of boolean arrays
    """

    # create output dataset if required
    if output_descriptor:
        driver_mem = gdal.GetDriverByName("MEM")
        output_tmp_name = os.path.join(tmpDir, 'tmp.mem')
        output_dataset = driver_mem.Create(output_tmp_name,
                                           img_to_classify.nx,
                                           img_to_classify.ny,
                                           4, gdal.GDT_Byte)
        output_v = np.ones((4, img_to_classify.ny, img_to_classify.nx), dtype=np.int8) * 255

    # use radiometric ranges to classify
    sets_colors = np.multiply(getColor(len(sets_rad_range)), 255)
    output_sets_def = []
    if type == 'slope':
        for idx in range(0, len(sets_rad_range)):
            if idx == len(sets_rad_range) - 1:
                output_sets_def.append(np.apply_along_axis(lambda x:(sets_rad_range[idx] <= x),
                                                           0, img_to_classify.r))
                if output_descriptor:
                    for i in range(0, 3):
                        output_v[i][sets_rad_range[idx] <= img_to_classify.r] = sets_colors[idx][i]
            else:
                output_sets_def.append(np.apply_along_axis(lambda x:(sets_rad_range[idx] <= x)*(x < sets_rad_range[idx+1]),
                                                           0, img_to_classify.r))
                if output_descriptor:
                    for i in range(0,3):
                        output_v[i][(sets_rad_range[idx] <= img_to_classify.r) *
                                    (img_to_classify.r < sets_rad_range[idx + 1])] = sets_colors[idx][i]

    # deals with the nan value (we choose black color for nan value since it is not part of the colormap chosen)
    if output_descriptor:
        for i in range(0,4):
            output_v[i][(np.isnan(img_to_classify.r)) + (img_to_classify.r == img_to_classify.nodata)] = \
                output_descriptor['nodata'][i]
    for idx in range(0, len(sets_rad_range)):
        output_sets_def[idx][(np.isnan(img_to_classify.r)) + (img_to_classify.r == img_to_classify.nodata)] = False

    # write down the result then translate from MEM to PNG
    if output_descriptor:
        [output_dataset.GetRasterBand(i + 1).WriteArray(output_v[i]) for i in range(0, 4)]
        gdal.GetDriverByName('PNG').CreateCopy(output_descriptor['path'], output_dataset)
        output_dataset = None

    return output_sets_def, sets_colors / 255

''''
# TODO a supp deplacé dans partition
def layers_fusion(clayers, sets, outputDir):
    """
    TODO Merge the layers to generate the layers fusion
    :param clayers: dict TODO
    :param sets: dict, mask by label for each layer
    :param outputDir: output directory
    :return: TODO
    """
    # TODO changer de nom et sortir de là
    def variables_activate(clayers, key_find):
        for k in clayers.keys():
            if not clayers[k][key_find]:
                return False
        return True

    # la fusion des layers (slope, map, ...) ne se fait que si toutes les layers sont renseignees (= 'reproject_[ref/dsm]' pas à None)
    all_layers_ref_flag = variables_activate(clayers, 'reproject_ref')
    all_layers_dsm_flag = variables_activate(clayers, 'reproject_dsm')

    dict_fusion = {'ref': all_layers_ref_flag, 'dsm': all_layers_dsm_flag}
    support_name = {'ref': 'Ref_support', 'dsm': 'DSM_support'}
    dict_stats_fusion = {'ref': None, 'dsm': None, 'reproject_ref': None, 'reproject_dsm': None,
                         'stats_results': {'Ref_support': None, 'DSM_support': None}}
    #dict_stats_fusion['stats_results'] = {'ref': None, 'dsm': None}
    classes_fusion = None
    all_combi_labels = None

    # create folder stats results fusion si layers_ref_flag ou layers_dsm_flag est à True
    if all_layers_ref_flag or all_layers_dsm_flag:
        create_stats_results(outputDir, 'fusion_layer')

        # Boucle sur [ref, dsm]
        #   Boucle sur chaque layer
        #       S'il y a plusieurs des layers données ou calculées
        #           ==> pour calculer les masks de chaque label
        #           ==> calculer toutes les combinaisons (developpement des labels entre eux mais pas les listes)
        #           ==> puis calculer les masks fusionnés (a associer avec les bons labels)
        #           ==> generer la nouvelle image classif (fusion) avec de nouveaux labels calculés arbitrairement et liés aux labels d entrees
        for df_k, df_v in dict_fusion.items():
            if df_v:
                # get les reproject_ref/dsm et faire une nouvelles map avec son dictionnaire associé
                clayers_to_fusion_path = [(k, clayers[k][str('reproject_' + df_k)]) for k in clayers.keys()]
                # lire les images clayers_to_fusion
                clayers_to_fusion = [(k, A3DGeoRaster(cltfp)) for k, cltfp in clayers_to_fusion_path]

                classes_to_fusion = []
                for k, cltfp in clayers.items():
                    classes_to_fusion.append([(k, cl_classes_label) for cl_classes_label in clayers[k]['classes'].keys()])

                if not (all_combi_labels and classes_fusion):
                    all_combi_labels, classes_fusion = create_new_classes(classes_to_fusion)

                # create la layer fusionnee + les sets assossiés
                sets_masks = sets[df_k]
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

    return clayers, sets_fusion
'''

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
            mask_label[sets_masks[layer_name]['sets_def'][label_name]] = 1
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


# TODO voir si a supp !!
def get_sets_labels_and_names(class_rad_range):
    """
    Get sets' labels and sets' names

    :param class_rad_range: list defining class ranges such as [0 10 25 100]
    :return sets labels and names
    """
    sets_label_list = []
    sets_name_list = []

    for i in range(0, len(class_rad_range)):
        if i == len(class_rad_range) - 1:
            sets_label_list.append(r'$\nabla$ > {}%'.format(class_rad_range[i]))
            sets_name_list.append('[{}; inf['.format(class_rad_range[i]))
        else:
            sets_label_list.append(r'$\nabla \in$ [{}% ; {}%['.format(class_rad_range[i], class_rad_range[i + 1]))
            sets_name_list.append('[{}; {}['.format(class_rad_range[i], class_rad_range[i + 1]))

    return sets_label_list, sets_name_list


# TODO voir si a supp !!
def get_sets_labels_and_names_for_classification(classes):
    """
    Get sets' labels and sets' names for classification_layer part

    :param classes: dict defining class labels and names
    :param support_ref: A3DGeoRaster classification reference
    :param support_dsm: A3DGeoRaster classification dsm
    :return: sets labels and names and classes updated
    """
    sets_label_list = list(classes.keys())
    if sets_label_list[0].find('[') == 0:
        sets_label_list = list()
        for label in list(classes.keys()):
            if label.find('inf]') > 0:
                new_label = '$\nabla$ > ' + label.split('[')[1].split(';inf]')[0]
            else:
                new_label = '$\nabla \in$ ' + label
            sets_label_list.append(new_label)

    sets_name_list = ['{} : {}'.format(key, value) for key, value in classes.items()]
    sets_name_list = [name.replace(',', ';') for name in sets_name_list]

    return sets_label_list, sets_name_list


def cross_class_apha_bands(ref_png_desc, dsm_png_desc, ref_sets, dsm_sets, tmpDir='.'):
    """
    Set accordingly the alpha bands of both png : for pixels where classification differs, alpha band is transparent

    :param ref_png_desc: dictionary with 'path' and 'nodata' keys for the ref support classified img (png format)
    :param dsm_png_desc: dictionary with 'path' and 'nodata' keys for the ref support classified img (png format)
    :param ref_sets: list of ref sets (ref_png class)
    :param dsm_sets: list of dsm sets (dsm_png class)
    :param tmpDir: where to store temporary data
    :return:
    """

    ref_dataset = gdal.Open(ref_png_desc['path'])
    dsm_dataset = gdal.Open(dsm_png_desc['path'])
    ref_mem_dataset = gdal.GetDriverByName('MEM').CreateCopy(os.path.join(tmpDir, 'tmp_ref.mem'), ref_dataset)
    dsm_mem_dataset = gdal.GetDriverByName('MEM').CreateCopy(os.path.join(tmpDir, 'tmp_sec.mem'), dsm_dataset)
    ref_aplha_v = ref_dataset.GetRasterBand(4).ReadAsArray()
    dsm_aplha_v = dsm_dataset.GetRasterBand(4).ReadAsArray()
    ref_dataset = None
    dsm_dataset = None

    # Combine pairs of sets together (meaning first ref set with first dsm set)
    # -> then for each single class / set, we know which pixels are coherent between both ref and dsm support img
    # -> combine_sets[0].shape[0] = number of sets (classes)
    # -> combine_sets[0].shape[1] = number of pixels inside a single DSM
    combine_sets = np.array([ref_sets[i][:] == dsm_sets[i][:] for i in range(0, len(ref_sets))])

    # Merge all combined sets together so that if a pixel's value across all sets is not always True then the alpha
    # band associated value is transparent (=0) since this pixel is not classified the same way between both support img
    # -> np.all gives True when the pixel has been coherently classified since its bool val were sets pairwise identical
    # -> np.where(...) gives indices of pixels for which cross classification is incoherent (np.all(...)==False)
    # -> those pixels as set transparent (=0) in chanel 4
    incoherent_indices=np.where(np.all(combine_sets,axis=0)==False)
    ref_aplha_v[incoherent_indices] = 0
    dsm_aplha_v[incoherent_indices] = 0

    # Write down the results
    ref_mem_dataset.GetRasterBand(4).WriteArray(ref_aplha_v)
    dsm_mem_dataset.GetRasterBand(4).WriteArray(dsm_aplha_v)

    # From MEM to PNG (GDAL does not seem to handle well PNG format)
    gdal.GetDriverByName('PNG').CreateCopy(ref_png_desc['path'], ref_mem_dataset)
    gdal.GetDriverByName('PNG').CreateCopy(dsm_png_desc['path'], dsm_mem_dataset)


def get_nonan_mask(array, nan_value):
    return np.apply_along_axis(lambda x: (~np.isnan(x))*(x != nan_value), 0, array)


def get_outliers_free_mask(array, no_data_value=None):
    if no_data_value:
        no_data_free_mask = get_nonan_mask(array, no_data_value)
    array_without_nan = array[np.where(no_data_free_mask == True)]
    mu = np.mean(array_without_nan)
    sigma = np.std(array_without_nan)
    return np.apply_along_axis(lambda x: (x > mu - 3 * sigma) * (x < mu + 3 * sigma), 0, array)


def create_mode_masks(alti_map, partitions_sets_masks=None):
    """
    Compute Masks for every required modes :
    -> the 'standard' mode where the mask stands for nan values inside the error image with the nan values
       inside the ref_support_desc when do_classification is on & it also stands for outliers free values
    -> the 'coherent-classification' mode which is the 'standard' mode where only the pixels for which both sets (dsm
       and reference) are coherent
    -> the 'incoherent-classification' mode which is 'coherent-classification' complementary

    Note that 'coherent-classification' and 'incoherent-classification' mode masks can only be computed if
    len(list_of_sets_masks)==2

    :param alti_map: A3DGeoRaster, alti differences
    :param partitions_sets_masks: [] (master and/or slave dsm) of [] of boolean array (sets for each dsm)
    :return: list of masks, associated modes, and error_img read as array
    """

    mode_names = []
    mode_masks = []

    # Starting with the 'standard' mask
    mode_names.append('standard')
    # -> remove alti_map nodata indices
    mode_masks.append(get_nonan_mask(alti_map.r, alti_map.nodata))
    # -> remove nodata indices for every partitioning image
    if partitions_sets_masks:
        for pImg in partitions_sets_masks:
            # for a given partition, nan values are flagged False for all sets hence
            # np.any return True for a pixel if it belongs to at least one set (=it this is not a nodata pixel)
            partition_nonan_mask = np.any(pImg, axis=0)
            mode_masks[0] *= partition_nonan_mask

    # Carrying on with potentially the cross classification (coherent & incoherent) masks
    if len(partitions_sets_masks) == 2:     # there's a classification img to partition from for both master & slave dsm
        mode_names.append('coherent-classification')
        # Combine pairs of sets together (meaning first partition first set with second partition first set)
        # -> then for each single class / set, we know which pixels are coherent between both partitions
        # -> combine_sets[0].shape[0] = number of sets (classes)
        # -> combine_sets[0].shape[1] = number of pixels inside a single DSM
        pImgs = partitions_sets_masks
        combine_sets = np.array([pImgs[0][set_idx][:] == pImgs[1][set_idx][:] for set_idx in range(0, len(pImgs[0]))])
        coherent_mask = np.all(combine_sets, axis=0)
        mode_masks.append(mode_masks[0] * coherent_mask)

        # Then the incoherent one
        mode_names.append('incoherent-classification')
        mode_masks.append(mode_masks[0] * ~coherent_mask)

    return mode_masks, mode_names


def create_masks(alti_map,
                 do_classification=False, ref_support=None,
                 do_cross_classification=False, ref_support_classified_desc=None,
                 remove_outliers = True):
    """
    Compute Masks for every required modes :
    -> the 'standard' mode where the mask stands for nan values inside the error image with the nan values
       inside the ref_support_desc when do_classification is on & it also stands for outliers free values
    -> the 'coherent-classification' mode which is the 'standard' mode where only the pixels for which both sets (dsm
       and reference) are coherent
    -> the 'incoherent-classification' mode which is 'coherent-classification' complementary

    :param alti_map: A3DGeoRaster, alti differences
    :param do_classification: boolean indicated wether or not the classification is activated
    :param ref_support: A3DGeoRaster
    :param do_cross_classification: boolean indicated wether or not the cross classification is activated
    :param ref_support_classified_desc: dict with 'path' and 'nodata' keys for the ref support image classified
    :param remove_outliers: boolean, set to True (default) to return a no_outliers mask
    :return: list of masks, associated modes, and error_img read as array
    """

    modes = []
    masks = []

    # Starting with the 'standard' mask with no nan values
    modes.append('standard')
    masks.append(get_nonan_mask(alti_map.r, alti_map.nodata))

    # Create no outliers mask if required
    no_outliers = None
    if remove_outliers:
        no_outliers = get_outliers_free_mask(alti_map.r, alti_map.nodata)

    # If the classification is on then we also consider ref_support nan values
    if do_classification:
        masks[0] *= get_nonan_mask(ref_support.r, ref_support.nodata)

    # Carrying on with potentially the cross classification masks
    if do_classification and do_cross_classification:
        modes.append('coherent-classification')
        ref_support_classified_dataset = gdal.Open(ref_support_classified_desc['path'])
        ref_support_classified_val = ref_support_classified_dataset.GetRasterBand(4).ReadAsArray()
        # so we get rid of what are actually 'nodata' and incoherent values as well
        coherent_mask = get_nonan_mask(ref_support_classified_val, ref_support_classified_desc['nodata'][0])
        masks.append(masks[0] * coherent_mask)

        # Then the incoherent one
        modes.append('incoherent-classification')
        masks.append(masks[0] * ~coherent_mask)

    return masks, modes, no_outliers


def stats_computation(array, list_threshold=None):
    """
    Compute stats for a specific array

    :param array: numpy array
    :param list_threshold: list, defines thresholds to be used for pixels above thresholds ratio computation
    :return: dict with stats name and values
    """
    if array.size:
        res = {
            'nbpts': array.size,
            'max':float(np.max(array)),
            'min': float(np.min(array)),
            'mean': float(np.mean(array)),
            'std': float(np.std(array)),
            'rmse': float(np.sqrt(np.mean(array*array))),
            'median': float(np.nanmedian(array)),
            'nmad': float(1.4826 * np.nanmedian(np.abs(array-np.nanmedian(array)))),
            'sum_err': float(np.sum(array)),
            'sum_err.err': float(np.sum(array * array)),
        }
        if list_threshold:
            res['ratio_above_threshold'] = {threshold: float(np.count_nonzero(array>threshold))/float(array.size)
                                            for threshold in list_threshold}
        else:
            res['ratio_above_threshold'] = {'none': np.nan}
    else:
        res = {
            'nbpts': array.size,
            'max': np.nan,
            'min': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'rmse': np.nan,
            'median': np.nan,
            'nmad': np.nan,
            'sum_err': np.nan,
            'sum_err.err': np.nan,
        }
        if list_threshold:
            res['ratio_above_threshold'] = {threshold: np.nan for threshold in list_threshold}
        else:
            res['ratio_above_threshold'] = {'none': np.nan}
    return res


def get_stats(dz_values, to_keep_mask=None, sets=None, sets_labels=None, sets_names=None,
              list_threshold=None):
    """
    Get Stats for a specific array, considering potentially subsets of it

    :param dz_values: errors
    :param to_keep_mask: boolean mask with True values for pixels to use
    :param sets: list of sets (boolean arrays that indicate which class a pixel belongs to)
    :param sets_labels: label associated to the sets
    :param sets_names: name associated to the sets
    :param list_threshold: list, defines thresholds to be used for pixels above thresholds ratio computation
    :return: list of dictionary (set_name, nbpts, %(out_of_all_pts), max, min, mean, std, rmse, ...)
    """
    def nighty_percentile(array):
        """
        Compute the maximal error for the 90% smaller errors

        :param array:
        :return:
        """
        if array.size:
            return np.nanpercentile(np.abs(array - np.nanmean(array)), 90)
        else:
            return np.nan

    # Init
    output_list = []
    nb_total_points = dz_values.size
    # - if a mask is not set, we set it with True values only so that it has no effect
    if to_keep_mask is None:
        to_keep_mask = np.ones(dz_values.shape)

    # Computing first set of values with all pixels considered -except the ones masked or the outliers-
    output_list.append(stats_computation(dz_values[np.where(to_keep_mask == True)], list_threshold))
    # - we add standard information for later use
    output_list[0]['set_label'] = 'all'
    output_list[0]['set_name'] = 'All classes considered'
    output_list[0]['%'] = 100 * float(output_list[0]['nbpts']) / float(nb_total_points)
    # - we add computation of nighty percentile (of course we keep outliers for that so we use dz_values as input array)
    output_list[0]['90p'] = nighty_percentile(dz_values[np.where(to_keep_mask==True)])

    # Computing stats for all sets (sets are a partition of all values)
    if sets is not None and sets_labels is not None and sets_names is not None:
        for set_idx in range(0,len(sets)):
            set = sets[set_idx] * to_keep_mask

            data = dz_values[np.where(set == True)]
            output_list.append(stats_computation(data, list_threshold))
            output_list[set_idx+1]['set_label'] = sets_labels[set_idx]
            output_list[set_idx+1]['set_name'] = sets_names[set_idx]
            output_list[set_idx+1]['%'] = 100 * float(output_list[set_idx+1]['nbpts']) / float(nb_total_points)
            output_list[set_idx+1]['90p'] = nighty_percentile(dz_values[np.where((sets[set_idx] * to_keep_mask) == True)])

    return output_list


def dem_diff_plot(dem_diff, title='', plot_file='dem_diff.png', display=False):
    """
    Simple img show after outliers removal

    :param dem_diff: A3DGeoRaster,
    :param title: string, plot title
    :param plot_file: path and name for the saved plot (used when display if False)
    :param display: boolean, set to True if display is on, otherwise the plot is saved to plot_file location
    :return:
    """

    #
    # Plot initialization
    #
    # -> import what is necessary for plot purpose
    import matplotlib as mpl
    mpl.rc('font', size=6)
    import matplotlib.pyplot as P

    #
    # Plot
    #
    P.figure(1, figsize=(7.0, 8.0))
    P.title(title)
    mu = np.nanmean(dem_diff.r)
    sigma = np.nanstd(dem_diff.r)
    P.imshow(dem_diff.r, vmin=mu-sigma, vmax=mu+sigma)
    cb = P.colorbar()
    cb.set_label('Elevation differences (m)')

    #
    # Show or Save
    #
    if display is False:
        P.savefig(plot_file, dpi=100, bbox_inches='tight')
    else:
        P.show()
    P.close()


def plot_histograms(input_array, bin_step=0.1, to_keep_mask=None,
                       sets=None, sets_labels=None, sets_colors=None,
                       plot_title='', outplotdir='.', outhistdir='.',
                       save_prefix='', display=False, plot_real_hist=False):
    """
    Creates a histogram plot for all sets given and saves them on disk.
    Note : If more than one set is given, than all the remaining sets are supposed to partitioned the first one. Hence
           in the contribution plot the first set is not considered and all percentage are computed in regards of the
           number of points within the first set (which is supposed to contain them all)

    :param input_array: data to plot
    :param bin_step: histogram bin step
    :param to_keep_mask: boolean mask with True values for pixels to use
    :param sets: list of sets (boolean arrays that indicate which class a pixel belongs to)
    :param set_labels: name associated to the sets
    :param sets_colors: color set for plotting
    :param sets_stats: where should be retrived mean and std values for all sets
    :param plot_title: plot primary title
    :param outplotdir: directory where histograms are to be saved
    :param outhistdir: directory where histograms (as numpy files) are to be saved
    :param save_prefix: prefix to the histogram files saved by this method
    :param dsplay: set to False to save plot instead of actually plotting them
    :param plot_real_hist: plot or save (see display param) real histrograms
    :return: list saved files
    """

    saved_files=[]
    saved_labels=[]
    saved_colors=[]

    #
    # Plot initialization
    #
    # -> import what is necessary for plot purpose
    import matplotlib as mpl
    mpl.rc('font', size=6)
    import matplotlib.pyplot as P
    from matplotlib import gridspec

    # -> bins should rely on [-A;A],A being the higher absolute error value (all histograms rely on the same bins range)
    if to_keep_mask is not None:
        borne = np.max([abs(np.nanmin(input_array[np.where(to_keep_mask==True)])),
                        abs(np.nanmax(input_array[np.where(to_keep_mask==True)]))])
    else:
        borne = np.max([abs(np.nanmin(input_array)), abs(np.nanmax(input_array))])
    bins = np.arange(-roundUp(borne, bin_step), roundUp(borne, bin_step)+bin_step, bin_step)
    np.savetxt(os.path.join(outhistdir, save_prefix+'bins'+'.txt'), [bins[0],bins[len(bins)-1], bin_step])

    # -> set figures shape, titles and axes
    #    -> first figure is just one plot of normalized histograms
    if plot_real_hist:
        P.figure(1, figsize=(7.0, 8.0))
        P.suptitle(plot_title)
        P.title('Data shown as normalized histograms')
        P.xlabel('Errors (meter)')
        data = []
        full_color = []
        for set_idx in range(0, len(sets)):
            # -> restricts to input data
            if to_keep_mask is not None:
                sets[set_idx] = sets[set_idx] * to_keep_mask
            data.append(input_array[np.where(sets[set_idx] == True)])
            full_color.append(sets_colors[set_idx])
        P.hist(data, density=True, label=sets_labels, histtype='step', color=full_color)
        P.legend()
        if display is False:
            P.savefig(os.path.join(outplotdir,'AltiErrors_RealHistrograms_'+save_prefix+'.png'),
                          dpi=100, bbox_inches='tight')
        P.close()
    #    -> second one is two plots : fitted by gaussian histograms & classes contributions
    P.figure(2, figsize=(7.0, 8.0))
    gs = gridspec.GridSpec(1,2, width_ratios=[10,1])
    P.subplot(gs[0])
    P.suptitle(plot_title)
    P.title('Errors fitted by a gaussian')
    P.xlabel('Errors (in meter)')
    P.subplot(gs[1])
    P.title('Classes contributions')
    P.xticks(np.arange(1), '')

    #
    # Plot creation
    #
    cumulative_percent = 0
    set_zero_size = 0
    if sets is not None and sets_labels is not None and sets_colors is not None:
        for set_idx in range(0,len(sets)):
            # -> restricts to input data
            if to_keep_mask is not None:
                sets[set_idx] = sets[set_idx] * to_keep_mask
            data = input_array[np.where(sets[set_idx] == True)]

            # -> empty data is not plotted
            if data.size:
                mean = np.mean(data)
                std = np.std(data)
                nb_points_as_percent = 100 * float(data.size) / float(input_array.size)
                if set_idx != 0:
                    set_contribution = 100 * float(data.size) / float(set_zero_size)
                    cumulative_percent += set_contribution
                else:
                    set_zero_size = data.size

                try:
                    n, bins = np.histogram(data, bins=bins, density=True)
                    popt, pcov = curve_fit(gaus, bins[0:bins.shape[0] - 1] + int(bin_step / 2), n,
                                           p0=[1, mean, std])
                    P.figure(2)
                    P.subplot(gs[0])
                    l = P.plot(np.arange(bins[0], bins[bins.shape[0] - 1], bin_step / 10),
                               gaus(np.arange(bins[0], bins[bins.shape[0] - 1], bin_step / 10), *popt),
                               color=sets_colors[set_idx], linewidth=1,
                               label=' '.join([sets_labels[set_idx],
                                               r'$\mu$ {0:.2f}m'.format(mean),
                                               r'$\sigma$ {0:.2f}m'.format(std),
                                               '{0:.2f}% points'.format(nb_points_as_percent)]))
                    if set_idx != 0:
                        P.subplot(gs[1])
                        P.bar(1, set_contribution, 0.05, color=sets_colors[set_idx],
                              bottom=cumulative_percent - set_contribution,
                              label='test')  # 1 is the x location and 0.05 is the width (label is not printed)
                        P.text(1, cumulative_percent - 0.5 * set_contribution, '{0:.2f}'.format(set_contribution),
                               weight='bold', horizontalalignment='left')
                except RuntimeError:
                    print('No fitted gaussian plot created as curve_fit failed to converge')
                    pass

                # save outputs (plot files and name of labels kept)
                saved_labels.append(sets_labels[set_idx])
                saved_colors.append(sets_colors[set_idx])
                saved_file = os.path.join(outhistdir, save_prefix + str(set_idx) + '.npy')
                saved_files.append(saved_file)
                np.save(saved_file, n)

    #
    # Plot save or show
    #
    P.figure(2)
    P.subplot(gs[0])
    P.legend(loc="upper left")
    if display is False:
        P.savefig(os.path.join(outplotdir,'AltiErrors-Histograms_FittedWithGaussians_'+save_prefix+'.png'),
                  dpi=100, bbox_inches='tight')
    else:
        P.show()

    P.figure(2)
    P.close()

    return saved_files, saved_colors, saved_labels


def save_results(output_json_file, stats_list, labels_plotted=None, plot_files=None, plot_colors=None, to_csv=False):
    """
    Saves stats into specific json file (and optionally to csv file)

    :param output_json_file: file in which to save
    :param stats_list: all the stats to save (one element per label)
    :param labels_plotted: list of labels plotted
    :param plot_files: list of plot files associdated to the labels_plotted
    :param plot_colors: list of plot colors associdated to the labels_plotted
    :param to_csv: boolean, set to True to save to csv format as well (default False)
    :return:
    """

    results = {}
    for stats_index, stats_elem  in enumerate(stats_list):
        results[str(stats_index)] = stats_elem
        if labels_plotted is not None and plot_files is not None and plot_colors is not None :
            if stats_elem['set_label'] in labels_plotted:
                try:
                    results[str(stats_index)]['plot_file'] = plot_files[labels_plotted.index(stats_elem['set_label'])]
                    results[str(stats_index)]['plot_color'] = tuple(plot_colors[labels_plotted.index(stats_elem['set_label'])])
                except:
                    print('Error: plot_files and plot_colors should have same dimension as labels_plotted')
                    raise

    with open(output_json_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    if to_csv:
        # Print the merged results into a csv file with only "important" fields and extended fieldnames
        # - create filename
        csv_filename = os.path.join(os.path.splitext(output_json_file)[0]+'.csv')
        # - fill csv_results with solely the filed required
        csv_results = collections.OrderedDict()
        for set_idx in range(0, len(results)):
            key = str(set_idx)
            csv_results[key] = collections.OrderedDict()
            csv_results[key]['Set Name'] = results[key]['set_name']
            csv_results[key]['% Of Valid Points'] = results[key]['%']
            csv_results[key]['Max Error'] = results[key]['max']
            csv_results[key]['Min Error'] = results[key]['min']
            csv_results[key]['Mean Error'] = results[key]['mean']
            csv_results[key]['Error std'] = results[key]['std']
            csv_results[key]['RMSE'] = results[key]['rmse']
            csv_results[key]['Median Error'] = results[key]['median']
            csv_results[key]['NMAD'] = results[key]['nmad']
            csv_results[key]['90 percentile'] = results[key]['90p']
        # - writes the results down as csv format
        with open(csv_filename, 'w') as csvfile:
            fieldnames = list(csv_results["0"].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)

            writer.writeheader()
            for set in csv_results:
                writer.writerow(csv_results[set])


def create_partitions(dsm, ref, outputDir, stats_opts):
    """
    Create or adapt all classification supports for the stats.
    If the support is a slope,it's transformed into a classification support.
    :param dsm: A3GDEMRaster, dsm
    :param ref: A3GDEMRaster, coregistered ref
    :param outputDir: ouput directory
    :param stats_opts: TODO
    :return: dict, with partitions information {'
    """
    to_be_clayers = stats_opts['to_be_classification_layers'].copy()
    clayers = stats_opts['classification_layers'].copy()

    logging.debug("list of to be classification layers: ", to_be_clayers)
    logging.debug("list of already classification layers: ", clayers)

    # Create obj partition
    partitions = []
    for layer_name, tbcl in to_be_clayers.items():
        partitions.append(Partition(layer_name, 'to_be_classification_layers', dsm, ref, outputDir, **tbcl))
    for layer_name, cl in clayers.items():
        partitions.append(Partition(layer_name, 'classification_layers', dsm, ref, outputDir, **cl))

    # Reproject every 'to_be_classification_layer' & 'classification_layer'
    [parti.rectify_map() for parti in partitions]

    # Create the fusion partition
    if len(partitions) > 1:
        try:
            partitions.append(Fusion_partition(partitions, outputDir))
        except NotEnoughDataToPartitionError:
            logging.info('Partitions could ne be created')
            pass

    [logging.debug("list of already classification layers: ", p) for p in partitions]
    for p in partitions:
        print("list of already classification layers: ", p)

    return partitions


def alti_diff_stats(cfg, dsm, ref, alti_map, display=False):
    """
    Computes alti error stats with graphics and tables support.

    If cfg['stats_opt']['class_type'] is not None those stats can be partitioned into different sets. The sets
    are radiometric ranges used to classify a support image. May the support image be the slope image associated
    with the reference DSM then the sets are slopes ranges and the stats are provided by classes of slopes ranges.

    Actually, if cfg['stats_opt']['class_type'] is 'slope' then computeStats first computes slope image and classify
    stats over slopes. If cfg['stats_opt']['class_type'] is 'user' then a user support image must be given to be
    classified over cfg['stats_opt']['class_rad_range'] intervals so it can partitioned the stats.

    When cfg['stats_opt']['class_type']['class_coherent'] is set to True then two images to classify are required
    (one associated with the reference DEM and one with the other one). The results will be presented through 3 modes:
    -standard mode,
    -coherent mode where only alti errors values associated with coherent classes between both classified images are used
    -and, incoherent mode (the coherent complementary one).

    :param cfg: config file
    :param dsm: A3GDEMRaster, dsm
    :param ref: A3GDEMRaster, coregistered ref
    :param alti_map: A3DGeoRaster, dsm - ref
    :param display: boolean, display option (set to False to save plot on file system)
    :return:
    """

    def get_title(cfg):
        # Set future plot title with bias and % of nan values as part of it
        title = ['MNT quality performance']
        dx = cfg['plani_results']['dx']
        dy = cfg['plani_results']['dy']
        biases = {'dx': {'value_m': dx['bias_value'], 'value_p': dx['bias_value'] / ref.xres},
                  'dy': {'value_m': dy['bias_value'], 'value_p': dy['bias_value'] / ref.yres}}
        title.append('(mean biases : '
                     'dx : {:.2f}m (roughly {:.2f}pixel); '
                     'dy : {:.2f}m (roughly {:.2f}pixel);'.format(biases['dx']['value_m'],
                                                                  biases['dx']['value_p'],
                                                                  biases['dy']['value_m'],
                                                                  biases['dy']['value_p']))
        rect_ref_cfg = cfg['alti_results']['rectifiedRef']
        rect_dsm_cfg = cfg['alti_results']['rectifiedDSM']
        title.append('(holes or no data stats: '
                     'Reference DSM  % nan values : {:.2f}%; '
                     'DSM to compare % nan values : {:.2f}%;'.format(100 * (1 - float(rect_ref_cfg['nb_valid_points'])
                                                                            / float(rect_ref_cfg['nb_points'])),
                                                                     100 * (1 - float(rect_dsm_cfg['nb_valid_points'])
                                                                            / float(rect_dsm_cfg['nb_points']))))
        return title

    def get_thresholds_in_meters(cfg):
        # If required, get list of altitude thresholds and adjust the unit
        list_threshold_m = None
        if cfg['stats_opts']['elevation_thresholds']['list']:
            # Convert thresholds to meter since all dem_compare elevation unit is "meter"
            original_unit = cfg['stats_opts']['elevation_thresholds']['zunit']
            list_threshold_m = [((threshold * u.Unit(original_unit)).to(u.meter)).value
                                for threshold in cfg['stats_opts']['elevation_thresholds']['list']]
        return list_threshold_m

    # Get outliers free mask (array of True where value is no outlier)
    outliers_free_mask = get_outliers_free_mask(alti_map.r, alti_map.nodata)

    # There can be multiple ways to partition the stats. We gather them all inside a list here:
    partitions = create_partitions(dsm, ref, cfg['outputDir'], cfg['stats_opts'])

    # For every partition get stats and save them as plots and tables
    for p in partitions:
        # Compute stats for each mode and every sets
        mode_stats, mode_masks, mode_names = get_stats_per_mode(alti_map,
                                                                sets_masks=p.sets_masks,
                                                                sets_labels=p.sets_labels,
                                                                sets_names=p.sets_names,
                                                                elevation_thresholds=get_thresholds_in_meters(cfg),
                                                                outliers_free_mask=outliers_free_mask)

        # Save stats as plots, csv and json and do so for each mode
        cfg['stats_results']['modes'][p.name] = save_as_graphs_and_tables(alti_map.r,
                                                                          p.out_dir,
                                                                          mode_masks,     # contains outliers_free_mask
                                                                          mode_names,
                                                                          mode_stats,
                                                                          p.set_masks,
                                                                          p.sets_labels,
                                                                          p.sets_colors,
                                                                          plot_title=''.join(get_title(cfg)),
                                                                          bin_step=cfg['stats_opts']['alti_error_threshold']['value'],
                                                                          display=display,
                                                                          plot_real_hist=cfg['stats_opts']['plot_real_hists'])


def save_as_graphs_and_tables(data_array, out_dir,
                              mode_masks, mode_names, mode_stats,
                              sets_masks, sets_labels, sets_colors,
                              plot_title='Title', bin_step=0.1, display=False, plot_real_hist=True):
    """

    :param data_array:
    :param out_dir:
    :param mode_masks:
    :param mode_names:
    :param mode_stats:
    :param sets_masks:
    :param sets_labels:
    :param sets_colors:
    :param plot_title:
    :param bin_step:
    :param display:
    :param plot_real_hist:
    :return:
    """

    mode_output_json_files = {}
    for mode in range(0, len(mode_names)):
        #
        # Create plots for the actual mode and for all sets
        #
        # -> we are then ready to do some plots !

        # TODO (peut etre prevoir une activation optionnelle du plotage...)
        sets_labels = ['all'] + sets_labels
        sets_colors = np.array([(0, 0, 0)] + list(sets_colors))
        plot_files = plot_histograms(data_array,
                                     bin_step=bin_step,
                                     to_keep_mask=mode_masks[mode],
                                     sets=[np.ones(data_array.shape, dtype=bool)] + sets_masks,
                                     sets_labels=sets_labels,
                                     sets_colors=sets_colors,
                                     plot_title='\n'.join(plot_title),
                                     outplotdir=os.path.join(out_dir,
                                                             get_out_dir('snapshots_dir')),
                                     # TODO pb dir avec slope et class
                                     outhistdir=os.path.join(out_dir,
                                                             get_out_dir('histograms_dir')),
                                     save_prefix=mode_names[mode],
                                     display=display,
                                     plot_real_hist=plot_real_hist)

        #
        # Save results as .json and .csv file
        #
        mode_output_json_files[mode_names[mode]] = os.path.join(out_dir,
                                                                get_out_dir('stats_dir'),
                                                                'stats_results_' + mode_names[mode] + '.json')
        save_results(mode_output_json_files[mode_names[mode]],
                     mode_stats[mode],
                     labels_plotted=sets_labels,
                     plot_files=plot_files,
                     plot_colors=sets_colors,
                     to_csv=True)

    return mode_output_json_files


def get_stats_per_mode(data, sets_masks=None, sets_labels=None, sets_names=None,
                       elevation_thresholds=None, outliers_free_mask=None):
    """
    Generates alti error stats with graphics and csv tables.

    Stats are computed based on support images which can be viewed as classification layers. The layers are represented
    by the support_sets which partitioned the alti_map indices. Hence stats are computed on each set separately.

    There can be one or two support_imgs, associated to just the same amount of supports_sets. Both arguments being
    lists. If two such classification layers are given, then this method also produces stats based on 3 modes:
    -standard mode,
    -coherent mode where only alti errors values associated with coherent classes between both classified images are used
    -and, incoherent mode (the coherent complementary one).

    :param data: array to compute stats from
    :param sets_masks: [] of one or two array (sets partitioning the support_img) of size equal to data ones
    :param sets_labels: sets labels
    :param sets_names: sets names
    :param elevation_thresholds: list of elevation thresholds
    :param outliers_free_mask:
    :return: stats, masks, names per mode
    """

    # Get mode masks and names (sets_masks will be cross checked if len(sets_masks)==2)
    mode_masks, mode_names = create_mode_masks(data, sets_masks)

    # Next is done for all modes
    mode_stats = []
    for mode in range(0, len(mode_names)):
        # Remove outliers
        if outliers_free_mask:
            mode_masks[mode] *= outliers_free_mask

        # Compute stats for all sets of a single mode
        mode_stats[mode] = get_stats(data.r,
                                     to_keep_mask=mode_masks[mode],
                                     sets=sets_masks[0],
                                     sets_labels=sets_labels,
                                     sets_names=sets_names,
                                     list_threshold=elevation_thresholds)

    return mode_stats, mode_masks, mode_names


def wave_detection(cfg, dh, display=False):
    """
    Detect potential oscillations inside dh

    :param cfg: config file
    :param dh: A3DGeoRaster, dsm - ref
    :return:

    """

    # Compute mean dh row and mean dh col
    # -> then compute the min between dh mean row (col) vector and dh rows (cols)
    res = {'row_wise': np.zeros(dh.r.shape, dtype=np.float32), 'col_wise': np.zeros(dh.r.shape, dtype=np.float32)}
    axis = -1
    for dim in list(res.keys()):
        axis += 1
        mean = np.nanmean(dh.r, axis=axis)
        if axis == 1:
            # for axis == 1, we need to transpose the array to substitute it to dh.r otherwise 1D array stays row array
            mean = np.transpose(np.ones((1, mean.size), dtype=np.float32) * mean)
        res[dim] = dh.r - mean

        cfg['stats_results']['images']['list'].append(dim)
        cfg['stats_results']['images'][dim] = copy.deepcopy(cfg['alti_results']['dzMap'])
        cfg['stats_results']['images'][dim].pop('nb_points')
        cfg['stats_results']['images'][dim]['path'] = os.path.join(cfg['outputDir'],
                                                                   get_out_file_path('dh_{}_wave_detection.tif'.format(dim)))

        georaster = A3DGeoRaster.from_raster(res[dim], dh.trans, "{}".format(dh.srs.ExportToProj4()), nodata=-32768)
        georaster.save_geotiff(cfg['stats_results']['images'][dim]['path'])
