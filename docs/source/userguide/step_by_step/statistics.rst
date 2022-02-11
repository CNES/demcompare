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

**Demcompare** offers two types of stats sets to define how the stats shall be classified to look at particular areas, and the statistics will then be displayed overall and by stats set:




1. To be classification layers
------------------------------

`to_be_classification_layers` are layers in which an input `ranges` list is used by **demcompare** to obtain a classification layer by classifying the slope within those ranges.


The default behavior in **demcompare** is to use the **slope** of both DEMs to classify the stats by the slope range *[0, 10, 25, 50, 90]*. The slope of each DEM is obtained as follows:

.. math::

    Slope_{DEM}(x,y) &= \sqrt{(gx / res_x)^2 + (gy / res_y)^2)} / 8


, where :math:`c_{gx}` and :math:`c_{gy}` are the result of the convolution :math:`c_{gx}=conv(DEM,kernel_x)` and :math:`c_{gy} = conv(DEM,kernel_y)` of the DEM with the kernels :


.. math::

    kernel_x = \begin{bmatrix}-1 & 0 & 1\\-2 & 0 & 2\\-1 & 0 & 1\end{bmatrix}


.. math::
    kernel_y = T(kernel_x)


The slope will then be classified by the `ranges` set with the `ranges` argument.

Each set will contain all the pixels for whom the **ref** slope is contained inside the associated slope range. At the end, there will be a stats set for each slope range.

A valid *to_be_classification_layers* configuration could be:

.. code-block:: bash

    "to_be_classification_layers": {"slope": {"ranges": [0, 5, 10, 25, 45]}}




2. Classification layers
------------------------

`classification_layers` are layers for which the user has a segmentation or semantic segmentation map where pixels are gathered inside superpixels and called a label value.

The user can set as many exogenous layers to classify the stats as he requires, for instance: land cover map, validity masks, etc.

For every exogenous layer, the user should specify the superimposable DEM. **ref** and **dsm** keywords are designed to register the path of the exogenous layer, respectively superimposable to the **ref** or the **sec**.

All of the classification layers will be used separately to classify the stats, and then be merged into a full classification layer that will also be used to classify the stats.

A valid *classification_layers* configuration value could be:

.. code-block:: bash

    "classification_layers": {"status": {"dsm": 'path_to_land_cover_associated_with_the_dsm',
                                         "classes": {"valid": [0], "KO": [1],"Land": [2], "NoData": [3], "Outside_detector": [4]}}}





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
