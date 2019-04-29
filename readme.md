# dem_compare

This python software aims at comparing two DSMs together, whether or not they share common format, projection system,
planimetric resolution, and altimetric unit.

The coregistration algorithm is based on the Nuth & Kääb universal coregistration method.

dem_compare provides a wide variety of standard metrics and allows one to classify the statistics. The default behavior
classifies the stats by slope ranges but one can provide any other data to classify the stats from.

A comparison report can be compiled as html or pdf documentation with statistics printed as tables and plots.

## Usage

Run the python script `dem_compare.py` with a json configuration file as unique
argument (see `test_config.json` as an example):

    python dem_compare.py test_config.json

The results can be observed with:

    firefox test_output/report_documentation/html/dem_compare_report.html &

#### Conventions

Inside the `json` file, the DEMs are referred to as inputDSM and inputRef. The last one is supposed to be the denser one.
Hence, if any resampling process must be done, it is the only DEM that shall be resampled. Plus, if the DEMs where to be
uncoregistered, then the inputRef geographical location is supposedly the correct one.

#### WGS84 and EGM96 references

The DEMs altitudes can rely on both ellipsoid and geoid references. However one shall use the `georef` parameter to set
the reference assigned to the DEMs (the two DEMs can rely on different references).

#### Altimetric unit

It is assumed both DEMs altimetric unit is meter. If otherwise, one shall use the `zunit` to set the actual altimetric
unit.

#### ROI definition

The processed Region of interest (ROI) is either defined by (1) either the image coordinates (x,y) of its top-left corner,
and its dimensions (w, h) in pixels as a python dictionary with 'x', 'y', 'w' and 'h' keys or (2) the geographical
coordinates of the projected image as tuple with (left, right, bottom, top) coordinates

In anyway, this is four numbers that ought to be given in the `json` configuration file.

The ROI refers to the tested DEM and will be adapted to the REF dem georef by dem_compare.py itself.

If no ROI definition is provided then DEMs raster are fully processed.

#### Tile processing

Note that a `tile_size` parameter can be set to compute dem comparison by tile. As dem_compare can load the full size
DEMs (if no ROI is provided) it can sometimes run out of memory and fail. To prevent this from happening one can set a
`tile_size` (in pixel) where a tile is assumed to be squared. Then dem_compare will divide DEMs according to `tile_size`
and process each tile independently. To merge every tile results into one, some optional post-processing steps are
provided by dem_compare (see next chapter).

#### step by step process (and the possibility to avoid the coregistration step)

dem_compare allows one to execute only a subset of the whole process. As such, a `--step` command line argument is
provided. It accepts values from `{coregistration,stats,report}` :

    [user@machine] $ python dem_compare.py
    usage: dem_compare.py [-h]
                          [--step {coregistration,stats,report,mosaic,merge_stats,merge_plots} [{coregistration,stats,report,mosaic,merge_stats,merge_plots} ...]]
                          [--debug] [--display]
                          config.json

All the steps but stats are optional, and dem_compare can start at any step as long as previously required steps have been launched.
This means that one can launch the report step only as long as the stats step has already been performed from a previous
dem_compare launch and the config.json remains the same.
Note that the coregistration step is not mandatory for stats and following steps as one can decide its DEMs are already
coregistered.


#### The parameters

Here is the list of the parameters and the associated default value when it exists:

    {
        "outputDir" :
        "inputDSM" : {  "path",
                        "zunit" : "meter",
                        "georef" : "WGS84",
                        "nodata" : }
        "inputRef" : {  "path",
                        "zunit" : "meter",
                        "georef" : "WGS84",
                        "nodata" : }}
        "plani_opts" : {    "coregistration_method" : "nuth_kaab",
                            "disp_init" : {"x": 0, "y": 0}},
        "stats_opts" : {    "class_type": "slope",
                            "class_rad_range": [0, 10, 25, 50, 90],
                            "cross_classification": False,
                            "elevation_thresholds" : {"list": [0.5,1,3], "zunit": "meter"}
    }

## Processing the outputs

#### The stats

dem_compare results display some basic statistics computed on the image on altitude differences (calles `errors` below)
that are listed here :
- the minimal and maximal values
- the mean and the standard deviation
- the median and the NMAD (see the References section)
- the '90 percentile' which is actually the 90% percentile of the absolute differences between errors and the mean(errors)
- and, the percentage of valid points (for whom the statistics have been computed)

Those stats are actually displayed by stats set (see next section).

#### The stats sets

Using dem_compare one can get a closer look on particular areas compared together. It is by setting the `class_type` value
and the `class_rad_range` of the stats options (`stats_opts`) that one can set how the stats shall be classified if at
all.

The default behavior is `class_type`: `slope` which means the stats are going to be classified by slope range. The slope
range can be set with the `class_rad_range` argument. Hence, the statistics will be displayed overall, and by stats set,
each set containing all the pixels for whom the inputRef slope is contained inside the associated slope range. Hence,
there is a stats set for each slope range, and all the stats set form a partition of the altitude differences image.

Now, one can decide not to classify the stats by slope range but to use instead any other exogenous data he posses. For
that purpose, one might set `class_type`: `user` and add a file path as value to the `class_support_ref` key. Still,
the `class_rad_range` argument is to be set.

The `class_rad_range` key requires a list as : `[0, 5, 10]`. Set like this, three sets will partitioned the stats :
(1) the `class_support_ref` pixels with radiometry inside [0, 5[, (2) the one with radiometry inside [5, 10[, and (3),
the ones inside [10, inf[. Note that if `class_type` is set to `slope` (default value), then no `class_support_ref` is
required and the slope image to classify stats from will be computed by dem_compare on the inputRef DEM.

#### The cross classification and the modes

Along with classifying the statistics, dem_compare can display those stats in three different modes where a mode is
actually a set of all the pixels of the altitude differences image.

By default, the `cross_classification` is set to `False`. Hence only the standard mode will be displayed. If, the
`cross_classification` was to be set to `True`, then the coherent and the incoherent (both forming a partition of the
standard one) modes would also be displayed.

The cross classification activation allows dem_compare to classify not only the pixels of the inputRef (which is performed
when `class_type` is not `None`) but also the pixels of the inputDSM. This will be done the same way, which means this
will be done considering the same `class_rad_range`. Plus, if `class_type` is `slope` (default), then the classification
of the inputDSM pixels rely on its slope values (computed by dem_compare itself). And if `class_type` is set to `user`,
then one has to set a full file path to the `class_support_dsm` key.

Now here is how the modes are defined :
1. the standard mode results simply on all on valid pixels. This means nan values but also ouliers and masked ones are
discarded. Note that the nan values can be originated from the altitude differences image and / or the reference support
image. The last one being the slope image or the image given by the user as value to the class_support_ref` key and
`class_type` is not None.

2. the coherent mode which is the standard mode where only the pixels for which input DEMs classifications are coherent.

3. the incoherent mode which is the coherent one complementary.

#### The elevation threshold

Using the `elevation_thresholds` parameter one can set a list of thresholds. Then for each threshold dem_compare will
compute the ratio  of pixels for which the altitude difference is larger than this particular threshold.

Note that so far results are only visible inside `stats_results-*.json` output files (see next chapter). Please also
note that the threshold is compared against the altitude differences being signed. This means that the result is not
always relevant and this stats computation shall be used carefully.

#### The dh map and the intermediate data

dem_compare will store several data and here is a brief explanation for each one.

First, the images :

1. the `intial_dh.tif` image is the altitude differences image when both DEMs have been reprojected to the same grid (the
one of inputDSM) and no coregistration has been performed.

2. `final_dh.tif` is the altitude differences image from the reprojected DEMs after the coregistration

3. the `coreg_DSM.tif` and `coreg_Ref.tif` are the coregistered DEMS.

4. the `Ref_support.tif` and `DSM_support.tif` are the images from which the stats have been classified. Depending on the
values given to the parameters those images might not be there. With default behavior only the `Ref_support.tif` is computed
and it is the `coreg_Ref.tif` slope. When `cross_classification` is on, and `class_type` is `slope` (default), then
`Ref_support.tif` and `DSM_support.tif` and both slope images. Plus, in this case, the `Ref_support-DSM_support.tif`
(which the slope differences between both slope images) is also computed and stored.

5. the `Ref_support_classified.png` and the `DSM_support_classified.png` are the classified version of the images listed
previously. The alpha band is used to mask the pixels for whom both classification do not match. This could be because
one pixel has a slope between [0; 20[ for one DEM and between [45; 100[ for the other one.

6. the images whose names start with 'AltiErrors-' are the plots saved by dem_compare. They show histograms by stats
set and same histograms fitted by gaussian.

Then, the remaining files :

7. the `final_config.json` is the completion of the initial `config.json` file given by the user. It contains additional
information and is used when dem_compare is launched step by step.

8. the files whose names start with 'stats_results-' are the `.json` and `.csv` files listed the statistics for each
set. There is one file by mode.

9. the `.npy` files are the numpy histograms for each stats mode and set.

Eventually, one shall find in the `report_documentation/ directory` the full documentation with all the results presented
for each mode and each set, in `html` or `latex` format.

## Dependencies

Here is the list of required dependencies for the python environment:

    `gdal` with version 2.1.0 or higher.
    `numpy`
    `scipy`
    `pyproj`
    `astropy`
    `matplotlib`

For the report to be compiled one shall install `sphinx` and `latex` (for the .pdf version).

## References

For more details about the NMAD metric :
Höhle, J., Höhle, M., 2009. *Accuracy assessment of Digital Elevation Models by means of robust statistical methods.*
 ISPRS Journal of Photogrammetry and Remote Sensing 64(4), 398-406.

For the Nuth & Kääb universal coregistration algorithm :
Nuth, C. Kääb, 2011. *A. Co-registration and bias corrections of satellite elevation data sets for quantifying glacier
thickness change.* Cryosphere 5, 271290.
