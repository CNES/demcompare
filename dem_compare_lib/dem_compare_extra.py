#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Copyright (C) 2017-2018 Centre National d'Etudes Spatiales (CNES)

"""
dem_compare_extra contains scripts unecessary for dem_compare but which can still add value to the results like:
- merge_stats that merges stats computed by dem_compare.py on several tiles / part of the image
- merge_plots that merges plots computed by dem_compare.py on several tiles / part of the image

Be aware that dem_compare_extra assumes it knows the dem_compare output files and tree. Hence, if it was somehow
altered because of a dem_compare.py evolution, then dem_compare_extra might need an evolution as well.
"""
from __future__ import print_function
import os
import sys
import shutil
import json
import argparse
import numpy as np
import matplotlib as mpl
from dem_compare_lib.a3d_georaster import A3DGeoRaster
from dem_compare_lib.output_tree_design import get_out_dir, get_out_file_path
from stats import create_sets, create_masks

DEFAULT_STEPS = []
ALL_STEPS = ['mosaic', 'merge_stats', 'merge_plots']


def load_json(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def computeMosaic(tiles_path, output_dir):
    """
    Compute mosaic thanks to s2p_mosaic.py from all images resulting from tiles computation

    :param tiles_path: a list of all the tiles path
    :param output_dir: directory where to store json and csv output files
    :return:
    """

    import mosaic

    # Reads .json files
    # - get rid of invalid tiles (the ones without a final_config.json file)
    final_json_file = 'final_config.json'
    valid_tiles_path = [tile_path for tile_path in tiles_path if os.path.isfile(os.path.join(tile_path,
                                                                                             get_out_file_path(final_json_file)))]
    # - load the tiles final config json files
    tiles_final_cfg = [load_json(os.path.join(a_valid_tile, final_json_file)) for a_valid_tile in valid_tiles_path]

    # all tiles do not have all images, they might have failed somewhere before processing remaining images
    #  -> get all image lists inside a single big list to own them all
    list_of_label_img_list = [config['stats_results']['images']['list'] if 'stats_results' in config else []
                              for config in tiles_final_cfg]
    #  -> get the biggest list (we strongly assume here that the biggest list contains all the images possibly created)
    index_biggest, biggest_label_img_list = max(enumerate(list_of_label_img_list), key=lambda tup: len(tup[1]))
    #  -> get the images' name corresponding to the list of labels
    img_list = [os.path.basename(tiles_final_cfg[index_biggest]['stats_results']['images'][label_img]['path'])
                for label_img in biggest_label_img_list]
    # -> we add the dzMap and the initial_dh
    for config in tiles_final_cfg:
        if 'alti_results' in config and 'dzMap' in config['alti_results']:
            img_list.append(os.path.basename(config['alti_results']['dzMap']['path']))
            continue
    img_list.append('initial_dh.tif')

    tiles = [os.path.join(a_valid_tile, final_json_file) for a_valid_tile in valid_tiles_path]
    for img in img_list:
        if os.path.splitext(img)[1] != '.png':
            output_img = os.path.join(output_dir, img)
            color = False
            nbBands = 1
            dataType='Float32'
        else:
            output_img = os.path.join(output_dir, os.path.splitext(img)[0]+'.tif')
            color = True
            nbBands = 4
            dataType = 'Byte'

        mosaic.main(tiles, output_img, img, color=color, nbBands=nbBands, dataType=dataType)


def computeMergePlots(tiles_path, output_dir):
    """
    Creates a summary plot. To achieve goog performance, fitted gaussian are directly deduced from the merge stats :
     - mean
     - std

    Hence no normalization of the gaussian is performed. The bins range is also deduced from the merge stats :
     - max absolute error

    As a consequence, if the merge_stats step has not been performed already, then it is going to be launched by
    computeMergePlots.

    :param tiles_path: a list of all the tiles path
    :param output_dir: directory where to store json and csv output files
    :return:
    """

    from stats import gaus

    #
    # Plot init
    #
    # we deactivate the display, we just want to save the plot in a plot file
    mpl.use('Agg')
    mpl.rc('font', size=6)
    import matplotlib.pyplot as P
    from matplotlib import gridspec

    #
    # Get the modes list
    #
    import glob
    prefix = os.path.join(output_dir, get_out_dir('stats_dir'), 'merged_stats_for_')
    suffix = '_mode.json'
    json_mode_files = glob.glob('{}*{}'.format(prefix, suffix))
    if len(json_mode_files) == 0:
        # If merge_stats has not been performed, we launch it so we can get access to merged mean and std
        computeMergeStats(tiles_path, output_dir)
        json_mode_files = glob.glob('{}*{}'.format(prefix, suffix))
    modes = [filename.replace(prefix,'').replace(suffix, '') for filename in json_mode_files]


    #
    # Because we are going to need some information inside the final_config.json we need to load them
    # - get rid of invalid tiles (the ones without a final_config.json file)
    final_json_file = 'final_config.json'
    valid_tiles_path = [tile_path for tile_path in tiles_path if os.path.isfile(os.path.join(tile_path,
                                                                                             get_out_file_path(final_json_file)))]

    #
    # Special case : only one valid tile => its results are copy / pasted
    #
    if len(valid_tiles_path) == 1:
        # Get the tile plots
        tile_plots = glob.glob(os.path.join(valid_tiles_path[0], 'AltiErrors-Histograms_*'))

        # Output plots
        output_plots = [os.path.join(output_dir, os.path.basename(plot)) for plot in tile_plots]

        # Copy plots
        [shutil.copyfile(tile_plot, output_plot) for output_plot, tile_plot in zip(output_plots, tile_plots)]
        return

    # - load the tiles final config json files
    tiles_final_cfg = [load_json(os.path.join(a_valid_tile, final_json_file)) for a_valid_tile in valid_tiles_path]
    # - compute the weight mean biases without using nan values (trying to be consistent with tiles used to merge stats)
    nb_valid_pts = np.nansum(np.array([cfg['alti_results']['dzMap']['nb_valid_points'] for cfg in tiles_final_cfg]))
    dx = np.nansum(np.array([cfg['plani_results']['dx']['bias_value'] * cfg['alti_results']['dzMap']['nb_valid_points']
                             for cfg in tiles_final_cfg])) / nb_valid_pts
    dy = np.nansum(np.array([cfg['plani_results']['dy']['bias_value'] * cfg['alti_results']['dzMap']['nb_valid_points']
                             for cfg in tiles_final_cfg])) / nb_valid_pts
    # - compute the percentage of valid points
    percent_valid_dsm_pts = 100 * \
                            np.nansum(np.array([cfg['alti_results']['rectifiedDSM']['nb_valid_points'] for cfg in tiles_final_cfg])) / \
                            np.nansum(np.array([cfg['alti_results']['rectifiedDSM']['nb_points'] for cfg in tiles_final_cfg]))
    percent_valid_ref_pts = 100 * \
                            np.nansum(np.array([cfg['alti_results']['rectifiedRef']['nb_valid_points'] for cfg in tiles_final_cfg])) / \
                            np.nansum(np.array([cfg['alti_results']['rectifiedRef']['nb_points'] for cfg in tiles_final_cfg]))
    # - get DSM resolution from one of its rectified georef tile
    dsm = A3DGeoRaster(str(tiles_final_cfg[0]['alti_results']['rectifiedDSM']['path']), load_data=False)

    # One plot by mode
    for mode in modes:
        # Prepare the figure with two subplots (one with the fitted gaussian, and one with the % of each stats set)
        P.figure(1, figsize=(7.0, 8.0))
        gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1])
        P.subplot(gs[0])
        P.title('Error fitted by a gaussian')
        P.xlabel('Errors [DSM - REF] (meter)')
        P.subplot(gs[1])
        P.title('Contributions')
        P.xlabel('Population')
        P.xticks(np.arange(1), '')

        # Read the associated json file
        stats = load_json('{}{}{}'.format(prefix, mode, suffix))

        # Get the max and min error values for xrange
        max_error = max([stats[set]['max'] for set in stats.keys()])
        min_error = min([stats[set]['min'] for set in stats.keys()])
        x_step = 0.01
        xvalues = np.arange(min_error, max_error, x_step)

        # The plots are done by stats set
        cumulative_percent = 0
        for set_idx in range(len(stats.keys())):
            # little workaround to make sure merge plots will present sets in same order as individual plots
            set = str(set_idx)
            # plots fitted gaussian
            P.subplot(gs[0])
            linestyle='-'
            if set == '0':
                # ! it is assumed that the first set is the complete / full / 'all' set and the others partition it
                linestyle = '--'
            l = P.plot(xvalues,
                       gaus(xvalues, *[(1/(stats[set]['std']*np.sqrt(2*np.pi))),stats[set]['mean'], stats[set]['std']]),
                       color=stats[set]['plot_color'], ls=linestyle, linewidth=1.5,
                       label=' '.join([stats[set]['set_label'],
                                       r'$\mu$ {0:.2f}m'.format(stats[set]['mean']),
                                       r'$\sigma$ {0:.2f}m'.format(stats[set]['std']),
                                      '{0:.2f}% points'.format(stats[set]['%'])]))

            # plots cumulative percentage
            if set != '0':
                # ! it is assumed that the first set is the complete / full / 'all' set and the others partition it
                set_contrib = 100.0 * float(stats[set]['nbpts']) / float(stats['0']['nbpts'])
                cumulative_percent += set_contrib
                P.subplot(gs[1])
                P.bar(1, set_contrib, 0.05, color=stats[set]['plot_color'],
                      bottom=cumulative_percent - set_contrib,
                      label='does not matter')  # 1 is the x location and 0.35 is the width
                P.text(1, cumulative_percent - 0.5 * set_contrib, '{0:.2f}'.format(set_contrib),
                       weight='bold', horizontalalignment='left')

        # now, once for all stats set, we add some information / title / legend
        P.subplot(gs[0])
        primaryTitle = '\n'.join(['MNT altitude quality performance considering ranges of slope',
                                  '(mean biases : dx : {:.2f}m (roughly {:.2f}pixel);'
                                  'dy : {:.2f}m (roughly {:.2f}pixel))'.format(dx,
                                                                               dy,
                                                                               dx/dsm.xres,
                                                                               dy/dsm.yres),
                                  '(Reference DSM % nan points : {:.2f}%; '
                                  'DSM to compare % nan points : {:.2f}%)'.format(percent_valid_ref_pts,
                                                                                  percent_valid_dsm_pts)])
        P.suptitle(primaryTitle)
        P.legend(loc="upper left")

        # limit xrange so we can actually see something relevant
        bin_min = min_error
        bin_max = max_error
        P.xlim([bin_min,bin_max])

        # save the figure
        P.savefig(os.path.join(output_dir, get_out_dir('snapshots_dir'), 'AltiErrors-Histograms_FittedWithGaussians_' + mode + '.png'),
                  dpi=100, bbox_inches='tight')


def _mergePercentile(a_final_config, tile_stats_list, the_zip, infimum, supremum, kth_element, mode='standard', p=0.5,
                     func=lambda x, *kwargs: x, arg=None):
    """
    Compute percentile of big data set distributed over several image file.

    :param a_final_config: a dem_compare final_config.json already read
    :param tile_stats_list: list of individual dem_compare stats (with individual median and number of valid points)
    :param the_zip: list of tuples, (image to compute percentile for, support image to classify the stats)
    :param infimum: dictionary, percentile infinum over tiles and by stats set (stats set are defined within a_final_config)
    :param supremum: dictionary, percentile supremum over tiles and by stats set (stats set are defined within a_final_config)
    :param kth_element: dictionary, the kth element (for each stats set) where to find the value of the percentile computed
    :param mode: the mode for which to compute the percentile
    :param p: the percentile to compute
    :param func: a function to apply to the image before computing the percentile for a given stats set
    :param arg: dictionary with stats set as key, argument to the func function
    :return: percentiles values as dictionary with stats set as keys
    """

    # We design here a function required to get all information needed for a single image
    def getSingleImageAllInformationRequired(img_name, support_img_name, a_final_config, mode):
        # open img data
        img = A3DGeoRaster(str(img_name), nodata=a_final_config['alti_results']['dzMap']['nodata'])
        all_sets = [np.ones(img.r.shape, dtype=bool)]

        # open support img and get the sets indexes (add them to all_sets)
        support_img = None
        do_classify_results = False
        if support_img_name:
            do_classify_results = True
            support_img = A3DGeoRaster(str(support_img_name),
                                       nodata=a_final_config['stats_results']['images']['Ref_support']['nodata'])
            # get the sets' indexes
            sets_idx, sets_color = create_sets(support_img, a_final_config['stats_opts']['class_rad_range'])
            all_sets = all_sets + sets_idx

        # get the mode mask
        if mode != 'standard':
            classified_support_img_descriptor = a_final_config['stats_results']['images']['Ref_support_classified']
            dirname = os.path.dirname(img_name)
            classified_support_img_name = os.path.basename(classified_support_img_descriptor['path'])
            classified_support_img_descriptor['path'] = os.path.join(dirname, classified_support_img_name)
            do_cross_classification = True
        else:
            classified_support_img_descriptor = None
            do_cross_classification = False
        modes_masks, modes, no_outliers_mask = create_masks(img, do_classify_results, support_img,
                                                            do_cross_classification,
                                                            classified_support_img_descriptor,
                                                            remove_outliers=True)
        full_mask = modes_masks[modes.index(mode)] * no_outliers_mask

        return img, full_mask, all_sets

    #
    # Start the computation of the percentile required
    #

    # If we dot not have infimum and supremum percentile values, we need to compute them
    if infimum is None or supremum is None:
        tiles_local_percentiles = []
        for img_name, support_img_name in the_zip:
            # ... get all information required (which is the points to keep for the mode and the sets)
            img, modemask, all_sets = getSingleImageAllInformationRequired(img_name, support_img_name, a_final_config, mode)

            # ... for the current image, suppress the points we do not need to keep for this mode
            img.r = img.r[np.where(modemask == True)]

            # ... then for each set
            local_percentiles = {stats_set: np.nan for stats_set in tile_stats_list[0]}
            for stats_set_id in range(len(all_sets)):
                stats_set_key = str(stats_set_id)

                # ... apply the func function to the image
                data = func(img.r, arg[stats_set_key] if arg is not None else None)

                # ... for the current set, suppress the points we do not need to keep for this mode
                all_sets[stats_set_id] = all_sets[stats_set_id][np.where(modemask == True)]

                # ... for the current image, suppress the points we do not need for this set
                data = data[np.where(all_sets[stats_set_id] == True)]

                # ... compute the p percentile for this tile and this set
                local_percentiles[stats_set_key] = np.nanpercentile(data, p*100)

            # ... store for this tile, the p percentiles obtained for each set
            tiles_local_percentiles.append(local_percentiles)

        # then, compute for each set the infimum and supremum considering all tiles
        infimum = {stats_set: np.nan for stats_set in tile_stats_list[0]}
        supremum = {stats_set: np.nan for stats_set in tile_stats_list[0]}
        for stats_set_key in tile_stats_list[0]:
            infimum[stats_set_key] = np.nanmin(np.array([local_percentiles[stats_set_key]
                                                         for local_percentiles in tiles_local_percentiles]))
            supremum[stats_set_key] = np.nanmax(np.array([local_percentiles[stats_set_key]
                                                          for local_percentiles in tiles_local_percentiles]))

    # For each image do...
    nb_before = {stats_set: 0 for stats_set in tile_stats_list[0]}
    sub_img = {stats_set: [] for stats_set in tile_stats_list[0]}
    for img_name, support_img_name in the_zip:
        # ... get all information required (which is the points to keep for the mode and the sets)
        img, modemask, all_sets = getSingleImageAllInformationRequired(img_name, support_img_name, a_final_config, mode)

        # ... for the current image, suppress the points we do not need to keep for this mode
        img.r = img.r[np.where(modemask == True)]

        # ... then for each set
        for stats_set_id in range(len(all_sets)):
            stats_set_key = str(stats_set_id)

            # ... apply the func function to the image
            data = func(img.r, arg[stats_set_key] if arg is not None else None)

            # ... for the current set, suppress the points we do not need to keep for this mode
            all_sets[stats_set_id] = all_sets[stats_set_id][np.where(modemask == True)]

            # ... find out how many values are lesser than infimum (after having filtered by mode and set masks)
            nb_before[stats_set_key] += data[np.where((data < infimum[stats_set_key])
                                                       * (all_sets[stats_set_id] == True))].size

            # ... only keep in memory the part of img between infimum and supremum
            sub_img[stats_set_key].append(data[np.where((data <= supremum[stats_set_key]) *
                                                        (data >= infimum[stats_set_key]) *
                                                        (all_sets[stats_set_id] == True))])

        img = None

    # Assuming all nb_before elements have been discarded from kth_element, find out how many elements remain
    remaining_elements = {stats_set: kth_element[stats_set] - nb_before[stats_set] for stats_set in tile_stats_list[0]}

    # Get the percentile from the concatenated sub images
    output_dict = {}
    for stats_set in tile_stats_list[0]:
        data = np.sort(np.concatenate(sub_img[stats_set]))
        if kth_element[stats_set] == nb_before[stats_set]:
            # no value remaining here inside [infimum; supremum]
            # this is special case where maybe infimum == supremum
            if infimum[stats_set] == supremum[stats_set]:
                if np.isnan(infimum[stats_set]):
                    stats_class_name = tile_stats_list[0][stats_set]['set_name']
                    print('WARNING : Could not compute the desired percentile for stats range named {}.'
                          'This is because no tile contains data for this stats range'.format(
                        stats_class_name))
                    output_dict[stats_set] = np.nan
                    continue
            stats_class_name = tile_stats_list[0][stats_set]['set_name']
            print('WARNING : The desired percentile for stats range named {} might be computed as an approximation of '
                  'the right value if more than a tile contains data for this stats range'.format(stats_class_name))
            output_dict[stats_set] = (infimum[stats_set] + supremum[stats_set])*0.5
        else:
            output_dict[stats_set] = float(data[[int(remaining_elements[stats_set])-1]])
    return output_dict

def _mergePercentiles(valid_tiles_path, tile_stats_list, mode='standard'):
    """
    Compute percentiles of big data set distributed over several image file.
    Returns the median, and the nmad


    :param valid_tiles_path: list of path of valid tiles where to find final dh maps one wishes to compute percentiles
    :param tile_stats_list: list of individual dem_compare stats (with individual median and number of valid points)
    :param mode: the mode for which to compute the percentiles
    :return: median, and nmad values all as dictionaries with stats set as keys
    """

    #
    # This method is a bit tricky... it computes p percentiles from distributed image files. In our case, these images
    # are the final dh map from several dem_compare launches.
    # But this is not where the tricky part belongs... because we need this p percentiles for some partition of these
    # images (say slope range for example). Now, to partition these images, we rely on some support images.
    # We have one of those for each final dh map. These support images are to be classified the way dem_compare did it
    # when computing the stats by stats sets (a stats set can be made out of some slope range). Then, _mergePercentiles
    # is going to compute one global p percentile for each stats set.
    # And (so that it is a bit trickier..), those p percentiles are to be computed for some 'mode'. The mode is one
    # of the dem_compare modes (standard, coherent-classification, incoherent-classification). Of course, depending on
    # the mode parameter, only a subdivision of the final dh map will be considered. But as we do not owe those
    # subdivision, we are going to compute them as well.
    # So here is how this is going to work :
    # -> there will be a single loop over the final dh map (img_list)
    #   -> for each img we will compute the mode list of indexes and the sets list of indexes
    #      (computing the mode / a set list of indexes means getting the list of indexes of the img to consider for this
    #       mode / set).
    # -> after the loop over the img_list is done, we will be able to compute the p percentile for all sets and the mode
    # Now hang on, we start :
    #

    #
    # First things first : we need to get the list of images (final dh map) and their support image
    #
    # read the json configuration file of a single tile to get the final dh map name and the support image name
    # WARNING : note that it is assumed here those names are the same for all tiles
    a_final_config = load_json(os.path.join(valid_tiles_path[0], get_out_file_path('final_config.json')))
    try:
        final_dh_filename = os.path.basename(a_final_config['alti_results']['dzMap']['path'])
        list_final_dh_files = [os.path.join(valid_tile_path, final_dh_filename) for valid_tile_path in valid_tiles_path]
        list_support_img_files = [None for valid_tile_path in valid_tiles_path]
        if a_final_config['stats_opts']['class_type']:
            support_img_filename = os.path.basename(a_final_config['stats_results']['images']['Ref_support']['path'])
            list_support_img_files = [os.path.join(valid_tile_path, support_img_filename)
                                                 for valid_tile_path in valid_tiles_path]
    except:
        raise
    # For convenience, we zip the final dh img with their respective classified support img
    the_zip = zip(list_final_dh_files, list_support_img_files)

    #
    # This part concerns the algorithm thought to compute the global percentiles.
    # Here, for each set, we get back
    #   - the median_min, the median_max, and the median kth_element to look for
    #
    median_min = {}
    median_max = {}
    kth_element = {}
    for stats_set in tile_stats_list[0]:
        medians = np.array([tile_results[stats_set]['median'] for tile_results in tile_stats_list])
        median_min[stats_set] = np.nanmin(medians)
        median_max[stats_set] = np.nanmax(medians)
        # Find out which kth_element (for the median) we are looking for based on the total number of valid points
        kth_element[stats_set] = 0.5 * np.nansum(np.array([tile_results[stats_set]['nbpts']
                                                           for tile_results in tile_stats_list]))
    # Then we compute the medians for all stats sets of the mode required
    medians = _mergePercentile(a_final_config, tile_stats_list, the_zip, median_min, median_max, kth_element, mode)

    #
    # We kind of repeat the same process but for the 'NMAD' now
    #   - Remember the 'NMAD' is defined as : 1.4826 * np.nanmedian(np.abs(array-np.nanmedian(array)))
    #     where array is the final dh map and so that np.nanmedian(array) is what we've just computed
    #   - So it is kind of like computed the overall median again except the data is not final_dh map but instead
    #     tmp_data = np.abs(final_dh - np.nanmedian(final_dh)).
    #
    # There is no conceivable way for us to compute the local medians of the tmp_data as it would require a lot of
    # processes (to classify the data along class_rand_range) that is actually already done inside _mergePercentile.
    # So, we will let _mergePercentile compute the local medians by telling it we just don't have them
    # n we compute the medians for all stats sets of the mode required
    res = _mergePercentile(a_final_config, tile_stats_list, the_zip, None, None, kth_element, mode,
                                func=lambda x,y:np.abs(x-y), arg=medians)
    nmads = {key: 1.4826 * value for key, value in res.items()}

    return medians, nmads


def computeMergeStats(tiles_path, output_dir, compute_percentile=True):
    """
    Merge the stats from previous independant dem_compare.py launches

    :param tiles_path: a list of all the tiles path
    :param output_dir: directory where to store json and csv output files
    :param compute_percentile: boolean, set to True to merge percentile as well
    :return:
    """

    from stats import save_results

    #
    # First we select only the valid tiles and we get back the modes to merge stats for
    #

    # get rid of invalid tiles (the ones without a final_config.json file)
    final_json_file = 'final_config.json'
    valid_tiles_path = [tile_path for tile_path in tiles_path if os.path.isfile(os.path.join(tile_path,
                                                                                             get_out_file_path(final_json_file)))]

    # read the json configuration file of a single tile to get the modes and the name of the .json associated
    # WARNING : note that it is assumed here that the stats .json file shared the same name for all the tiles !
    a_final_config = load_json(os.path.join(valid_tiles_path[0], final_json_file))
    try:
        modes = a_final_config['stats_results']['modes']
    except:
        raise

    #
    # Special case : only one valid tile => its results are copy / pasted
    #
    if len(valid_tiles_path) == 1:
        for mode in modes:
            # Get the json and csv stat filename
            the_json_name_for_this_mode = os.path.join(valid_tiles_path[0], os.path.basename(modes[mode]))
            the_csv_name_for_this_mode = the_json_name_for_this_mode.replace('.json', '.csv')

            # Output name
            json_output_name = os.path.join(output_dir, get_out_dir('stats_dir'), 'merged_stats_for_{}_mode.json'.format(mode))
            csv_output_name = json_output_name.replace('.json', '.csv')
            shutil.copyfile(the_json_name_for_this_mode, json_output_name)
            shutil.copyfile(the_csv_name_for_this_mode, csv_output_name)
        return

    #
    # Merge the stats by mode
    #
    for mode in modes:
        # Get the json stat filename
        the_json_name_for_this_mode = os.path.basename(modes[mode])

        # Create the list of stats json file
        list_result_json_files = [os.path.join(valid_tile_path, the_json_name_for_this_mode) for valid_tile_path in valid_tiles_path]

        # Read the stats json file for each tile
        #   - each file contains a dictionary with sets (or labels if you prefer) as keys, and stats as values
        #   - so what we have there is a list of dictionaries with sets as keys and sets stats as values
        all_results = [load_json(a_json_file) for a_json_file in list_result_json_files]

        #
        # Below, stats are obviously treated explicitly
        #

        # Merge all the stats dictionary into a single one that will work as a summary
        list_of_merged_results = []
        for set_idx in range(len(all_results[0])):
            # first we compute trivial data needed later
            key=str(set_idx)
            numberOfValidPoints = np.nansum(np.array([tile_results[key]['nbpts'] for tile_results in all_results]))
            numberOfPoints = np.nansum(np.array([100 * tile_results[key]['nbpts'] / tile_results[key]['%']
                                                 for tile_results in all_results if tile_results[key]['%']]))
            sum_err = np.nansum(np.array([tile_results[key]['sum_err'] for tile_results in all_results]))
            sum_errxerr = np.nansum(np.array([tile_results[key]['sum_err.err'] for tile_results in all_results]))

            # then we start by unconditional information
            merged_results = {}
            merged_results['set_name'] = all_results[0][key]['set_name']
            merged_results['set_label'] = all_results[0][key]['set_label']
            merged_results['90p'] = np.nan
            merged_results['nmad'] = np.nan
            merged_results['median'] = np.nan
            merged_results['plot_file'] = None
            if 'plot_color' in all_results[0][key]:
                merged_results['plot_color'] = all_results[0][key]['plot_color']

            # then we carry on with conditional stats
            if numberOfPoints:
                merged_results['nbpts'] = numberOfValidPoints
                merged_results['%'] = 100 * numberOfValidPoints / numberOfPoints
                merged_results['max'] = np.nanmax(np.array([tile_results[key]['max']
                                                                 for tile_results in all_results]))
                merged_results['min'] = np.nanmin(np.array([tile_results[key]['min']
                                                                 for tile_results in all_results]))
                merged_results['sum_err'] = sum_err
                merged_results['sum_err.err'] = sum_errxerr
                merged_results['mean'] = sum_err / numberOfValidPoints
                merged_results['std'] = np.sqrt((sum_errxerr / numberOfValidPoints) -
                                                           (merged_results['mean'] * merged_results['mean']))
                merged_results['rmse'] = np.sqrt(sum_errxerr / numberOfValidPoints)

                #NB : for above threshold ratio we do for every ratio :
                #       SUM(tile[i][ratio] * tile[i][nbPts]) / SUM(tile[i][nbPts])  with tile[i] a tile among all tiles
                merged_results['ratio_above_threshold'] = {threshold: np.nansum(np.array([tile_results[key]['ratio_above_threshold'][threshold]*tile_results[key]['nbpts']
                                                                                          for tile_results in all_results]))
                                                                      / merged_results['nbpts']
                                                           for threshold in all_results[0][key]['ratio_above_threshold']}
            else:
                merged_results['%'] = 0.0
                merged_results['nbpts'] = numberOfPoints
                merged_results['sum_err'] = np.nan
                merged_results['sum_err.err'] = np.nan
                merged_results['max'] = np.nan
                merged_results['min'] = np.nan
                merged_results['mean'] = np.nan
                merged_results['std'] = np.nan
                merged_results['rmse'] = np.nan
                merged_results['ratio_above_threshold'] = {threshold: np.nan
                                                           for threshold in all_results[0][key]['ratio_above_threshold']}
            list_of_merged_results.append(merged_results)

        # if required, we also compute merged percentiles
        if compute_percentile:
            means = {str(id): list_of_merged_results[id]['mean'] for id in range(len(list_of_merged_results))}
            medians, nmads = _mergePercentiles(valid_tiles_path, all_results, mode=mode)
            for set_idx in range(len(list_of_merged_results)):
                list_of_merged_results[set_idx]['median'] = medians[str(set_idx)]
                list_of_merged_results[set_idx]['nmad'] = nmads[str(set_idx)]

        # Now we can call the save_results method from stats.py to save results as json file and csv file in the
        # dem_compare style
        save_results(os.path.join(output_dir, 'merged_stats_for_{}_mode.json'.format(mode)),
                     list_of_merged_results, None, None, None, True)


def computeInitialization(config_json):
    """
    Initialize the process

    :param config_json:
    :return: config as a dictionary and the list of tiles path
    """
    # read the json configuration file
    if isinstance(config_json, dict):
        cfg = config_json
    else:
        with open(config_json, 'r') as f:
            cfg = json.load(f)

    # if no 'json_list_file' then nothing to do
    try:
        tiles_list = cfg['tiles_list_file']
        with open(tiles_list, 'r') as f:
            list_of_tiles_path = f.readlines()
        list_of_tiles_path = [tile_path.strip('\n') for tile_path in list_of_tiles_path]
        # get rid off "config.json" suffix if there
        list_of_tiles_path = [tile_path.split('config.json')[0] for tile_path in list_of_tiles_path]
    except:
        print("One shall indicate where to find the list of tiles from dem_compare.py previous launches to be merged. "
                        "Use the \'tiles_list_file\' key for this purpose.")
        raise

    # there must be a outputDir location specified
    try:
        outputDir = cfg['outputDir']
        from initialization import mkdir_p
        mkdir_p(outputDir)
    except:
        print("One might set a outputDir directory")
        raise

    return cfg, list_of_tiles_path

def main(json_file, steps=DEFAULT_STEPS, debug=False, force=False):
    #
    # Initialization
    #
    cfg, list_of_tiles_path = computeInitialization(json_file)
    sys.stdout.flush()

    #
    # Launches steps one by one
    #
    if 'merge_stats' in steps:
        computeMergeStats(list_of_tiles_path, cfg['outputDir'])
    if 'merge_plots' in steps:
        computeMergePlots(list_of_tiles_path, cfg['outputDir'])
    if 'mosaic' in steps:
        computeMosaic(list_of_tiles_path, cfg['outputDir'])


def get_parser():
    """
    ArgumentParser for dem_compare_extra

    :param None
    :return parser
    """
    parser = argparse.ArgumentParser(description=('dem_compare extra services'))

    parser.add_argument('config', metavar='config.json',
                        help=('path to a json file containing parameters'))
    parser.add_argument('--step', type=str, nargs='+', choices=ALL_STEPS,
                        default=DEFAULT_STEPS)
    parser.add_argument('--debug', action='store_true')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.config, args.step, debug=args.debug)
