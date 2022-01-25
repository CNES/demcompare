.. _statistics:

Statistics
==========

**Demcompare**'s results display some altitude differences **statistics**:

- **Minimal and maximal values**
- **Mean and the standard deviation**
- **Median and the NMAD** [NMAD]_
- **'90 percentile'**: 90% percentile of the absolute differences between errors and the mean (errors)
- **Percentage of valid points** (for whom the statistics have been computed)

Those stats are actually displayed by stats sets (see next section).

The stats sets
**************

**Demcompare** offers two types of stats sets to define how the stats shall be classified to look at particular areas compared together, and the statistics will then be displayed overall and by stats set :




To be classification layers
---------------------------

`to_be_classification_layers` are layers (exogenous rasters) that could be used as classification layers by the use of a `ranges` list. Hence, the slope layer **demcompare** computes itself belongs to this category.



- Slope default classification

    The default behavior in **demcompare** is to use the **slope** of both DEMs to classify the stats by slope range. The slope range can be set with the `ranges` argument. Hence, the slope layer **demcompare** computes itself belongs to the *to_be_classification_layers* category.\

    Each set will contain all the pixels for whom the **ref** slope is contained inside the associated slope range.

    At the end, there is a stats set for each slope range and all the stats set form a partition of the altitude differences image in `final_dh.tif` file.


Classification layers
---------------------

`classification_layers` are layers (exogenous raster) such as segmentation or semantic segmentation for which pixels are gathered inside superpixels whose values are shared by every pixels in a superpixel and called a label value.

The user can set as many exogenous layers to classify the stats from: land cover map, validity masks, etc.
All of them will be used separately to classify the stats, and then be merged into a full classification layer that will also be used to classify the stats
(in that case **demcompare** could display the results for 'elevated roads' for which pixels are 'valid pixels').

A valid classification_layers value could be:
                            "classification_layers": {"land_cover": {"ref": 'None_or_path_to_land_cover_associated_with_the_ref',
                                                                     "dsm": 'None_or_path_to_land_cover_associated_with_the_dsm',
                                                                     "classes": {"forest": [31, 32], "urbain": [42]}}}
    }


For every exogenous layer (example above `land_cover`), the user ought to specify each superimposable DEM. **ref** and **dsm** keywords are then designed to register the path of the exogenous layer, respectively superimposable to the **ref** or the **sec**.




The cross classification and the modes
**************************************

As shown in previous section, **demcompare** will classify stats according to classification layers (computed slopes or exogenous data provided by the user).
Along with classifying the statistics, **demcompare** displays the each of the stats sets in three different modes. A **mode** is
a set of all the pixels of the altitude differences image.

Now here is how the modes are defined:

1. The **standard mode** results on all valid pixels.

 - This means nan values but also outliers (if `remove_outliers` was set to True) and masked ones are discarded. Note that the nan values can be originated from the altitude differences image and / or the exogenous classification layers themselves.

2. The **coherent mode** is the standard mode where only the pixels sharing the same label for both DEMs classification layers are kept.

 - Say after a coregistration, a pixel *P* is associated to a 'grass land' inside a `ref` classification layer named `land_cover` and a `road` inside the `dsm` classification layer also named `land_cover`, then *P* is not coherent for **demcompare**.

3. The **incoherent mode** which is the coherent one complementary.

The elevation thresholds (Experimental)
***************************************

This functionality allows **demcompare** to compute the ratio  of pixels for which the altitude difference is larger than a particular given threshold.

One can configure the `elevation_thresholds` parameter with a list of thresholds.

.. note::  So far results are only visible inside `stats_results-*.json` output files (see next chapter). Please also note that the threshold is compared against the altitude differences being signed. This means that the result is not always relevant and this stats computation shall be used carefully.



For more details about the NMAD metric :

.. [NMAD] Höhle, J., Höhle, M., 2009. Accuracy assessment of Digital Elevation Models by means of robust statistical methods. ISPRS Journal of Photogrammetry and Remote Sensing 64(4), 398-406.
