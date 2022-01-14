.. _statistics:

Statistics
==========

Demcompare's results display some altitude differences **statistics**:

- **Minimal and maximal values**
- **Mean and the standard deviation**
- **Median and the NMAD** [NMAD]_
- **'90 percentile'**: 90% percentile of the absolute differences between errors and the mean (errors)
- **Percentage of valid points** (for whom the statistics have been computed)

Those stats are actually displayed by stats sets (see next section).

The stats sets
**************

Demcompare can set how the stats shall be classified to look at particular areas compared together with `to_be_classification_layers`
and / or the `classification_layers` parameters in the `stats_opts` stats section.

Slope default classification
****************************

The default behavior is to use the **slope** of both DEMs to classify the stats by slope range. The slope range can be set with the `ranges` argument. \
The statistics will then be displayed overall and by stats set.

Each set will contain all the pixels for whom the inputRef slope is contained inside the associated slope range.

At the end, there is a stats set for each slope range and all the stats set form a partition of the altitude differences image in `final_dh.tif` file.

Other classifications
*********************

It is also possible not to classify the stats by slope range but to use instead any other exogenous data. \
For that purpose, `to_be_classification_layers` and / or the `classification_layers` parameters have to be used:

- `to_be_classification_layers` are layers (exogenous rasters) that could be use as classification layers by the use of a `ranges` list. Hence, the slope layer Demcompare computes itself belongs to this category.
- `classification_layers` are layers (exogenous raster) such as segmentation or semantic segmentation for which pixels are gathered inside superpixels whose values are shared by every pixels in a superpixel and called a label value.

For every exogenous layer (example above `land_cover`), the user ought to specify each superimposable DEM. **ref** and **dsm** keywords are then  designed to register the path of the exogenous layer, respectively superimposable to the **ref** or the **dsm**.

The user can set as many exogenous layers to classify the stats from: land cover map, validity masks, etc.
All of them will be used separately to classify the stats, and then be merged into a full classification layer that will also be used to classify the stats
(in that case Demcompare could display the results for 'elevated roads' for which pixels are 'valid pixels').

The cross classification and the modes
**************************************

Along with classifying the statistics, Demcompare can display the stats in three different modes. A **mode** is
a set of all the pixels of the altitude differences image.

As shown in previous section, Demcompare will classify stats according to classification layers (computed slopes or exogenous data provided by the user).
For each classification layer, Demcompare knows if it is superimposable to the **ref** or the **dsm** to be evaluated. Now one could provides two land cover classification layers to Demcompare.
One that would come with the **ref** DEM. And one that would come with the **dsm**. In this case, Demcompare provides a three modes stats display.

Now here is how the modes are defined:

1. The **standard mode** results on all valid pixels.

 - This means nan values but also outliers (if `remove_outliers` was set to True) and masked ones are discarded. Note that the nan values can be originated from the altitude differences image and / or the exogenous classification layers themselves.

2. The **coherent mode** is the standard mode where only the pixels sharing the same label for both DEMs classification layers are kept.

 - Say after a coregistration, a pixel *P* is associated to a 'grass land' inside a `ref` classification layer named `land_cover` and a `road` inside the `dsm` classification layer also named `land_cover`, then *P* is not coherent for Demcompare.

3. The **incoherent mode** which is the coherent one complementary.

The elevation thresholds (Experimental)
***************************************

The tool allows to configure `elevation_thresholds` parameter with a list of thresholds.

For each threshold, Demcompare will compute the ratio  of pixels for which the altitude difference is larger than this particular threshold.

.. note::  So far results are only visible inside `stats_results-*.json` output files (see next chapter). Please also note that the threshold is compared against the altitude differences being signed. This means that the result is not always relevant and this stats computation shall be used carefully.



For more details about the NMAD metric :

.. [NMAD] Höhle, J., Höhle, M., 2009. Accuracy assessment of Digital Elevation Models by means of robust statistical methods. ISPRS Journal of Photogrammetry and Remote Sensing 64(4), 398-406.
