<h4 align="center">DEMcompare, a DEM comparison tool  </h4>

[![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0/)

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#install">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#outputs-processing">Outputs processing</a> •
  <a href="#references">References</a>
</p>

## Overview

This python software aims at **comparing two DEMs** together.

A DEM is a 3D computer graphics representation of elevation data to represent terrain.

**DEMcompare** has several characteristics:

* provides a wide variety of standard metrics and allows one to classify the statistics.
* works whether or not the two DEMs share common format, projection system,
planimetric resolution, and altimetric unit.
* the coregistration algorithm is based on the Nuth & Kääb universal coregistration method.
* the default behavior classifies the stats by slope ranges but one can provide any other data to classify the stats.
* A comparison report can be compiled as html or pdf documentation with statistics printed as tables and plots.

Only **Linux Plaforms** are supported (virtualenv or bare machine) with **Python 3** installed.


## Install

The default install mode is via the pip package, typically through a [virtualenv](https://docs.python.org/3/library/venv):

```
python3 -m venv venv
source venv/bin/activate
pip install demcompare
```

#### Developer mode
This package can be installed through the following commands:
```
    git clone https://github.com/CNES/demcompare
    cd demcompare
    make install
    source venv/bin/activate # to go in installed dev environment
```
Dependencies : **git**, **make**

Be careful: **Rasterio** have its own embedded version of **GDAL** \
Please use rasterio no-binary version in Makefile `install` if you want to use a GDAL local version:
```
python3 -m pip --no-cache-dir install --no-binary rasterio rasterio
```

#### Global autocompletion
If global autocompletion is not configured in your environment, please use the following command to have completion:
```
eval "$(register-python-argcomplete demcompare)"
```
See [Argcomplete documentation](https://kislyuk.github.io/argcomplete/#global-completion) for more details.

#### PDF generation
By default, html is available only: PDF report is not generated.
You need to install Latex environment for Sphinx to be able to use latexpdf.

See [Specific Sphinx LatexBuilder documentation](https://www.sphinx-doc.org/en/master/usage/builders/index.html#sphinx.builders.latex.LaTeXBuilder)


## Usage

Run the python script **demcompare** with a json configuration file as unique
argument (see [`tests/test_config.json`](./tests/test_config.json) as an example):
```
    cd tests/
    demcompare test_config.json
```
The results can be observed with:
```
    firefox test_output/doc/published_report/html/demcompare_report.html &
```

#### Conventions

Inside the `json` file, the DEMs are referred to as **inputDSM** and **inputRef**.

Note: **inputRef** DEM is supposed to be the denser one:
- if any resampling process must be done, only **inputRef** DEM shall be resampled.
- if the DEMs are uncoregistered, then the **inputRef** geographical location is supposedly the correct one.

#### WGS84 and EGM96 references

The DEMs altitudes can rely on both **ellipsoid** and **geoid** references. \
However one shall use the `georef` parameter to set
the reference assigned to the DEMs (the two DEMs can rely on different references).

#### Altimetric unit

It is assumed both DEMs altimetric unit is **meter**.\
 If otherwise, one shall use the `zunit` to set the actual altimetric
unit.

#### ROI definition

The processed Region of interest (ROI) is either defined by
1. either the image coordinates (x,y) of its top-left corner,
and its dimensions (w, h) in pixels as a python dictionary with 'x', 'y', 'w' and 'h' keys
2. **or** the geographical
coordinates of the projected image as tuple with (left, bottom, right, top) coordinates

In anyway, this is four numbers that ought to be given in the `json` configuration file.

The ROI refers to the tested DEM and will be adapted to the REF dem georef by demcompare itself.

If no ROI definition is provided then DEMs raster are fully processed.

#### Tile processing

Tile processing is not available anymore. A future version might provide a better way to deal with very large data. In
the meantime one can deal with heavy DSMs by setting a ROI (see previous chapter).

#### Step by step processing

`demcompare` allows one to execute only a subset of the whole process. As such, a `--step` command line argument is
provided. It accepts values from `{coregistration, stats, report}` :

    [user@machine] $ demcompare
    usage: demcompare [-h]
      [--step {coregistration,stats,report} [{coregistration,stats,report} ...]]
      [--display] [--version]
      config.json

- All the steps but **stats** are optional
- `demcompare` can start at any step as long as previously required steps have been launched.
  - This means that one can launch the report step only as long as the stats step has already been performed from a previous
`demcompare` launch and the config.json remains the same.
- **coregistration** step is not mandatory for stats and following steps as one can decide its DEMs are already coregistered.


#### The parameters

Here is the list of the parameters and the associated default value when it exists:

    {
        "outputDir" : "./test_output/"
        "inputDSM" : {  "path", "./inputDSM.tif"
                        "zunit" : "meter",
                        "georef" : "WGS84",
                        "nodata" : }
        "inputRef" : {  "path", "inputRef.tif"
                        "zunit" : "meter",
                        "georef" : "WGS84",
                        "nodata" : }}
        "plani_opts" : {    "coregistration_method" : "nuth_kaab",
                            "disp_init" : {"x": 0, "y": 0}},
        "stats_opts" : {    "elevation_thresholds" : {"list": [0.5,1,3], "zunit": "meter"},
                            "remove_outliers": False,
                            "to_be_classification_layers": {"slope": {"ranges": [0, 10, 25, 50, 90],
                                                                      "ref": None,
                                                                      "dsm": None}},
                            "classification_layers": {}
    }

Where a valid `classification_layers` value could be:

```
                            "classification_layers": {"land_cover": {"ref": 'None_or_path_to_land_cover_associated_with_the_ref',
                                                                     "dsm": 'None_or_path_to_land_cover_associated_with_the_dsm',
                                                                     "classes": {"forest": [31, 32], "urbain": [42]}}}
    }
```

## Outputs processing

#### The stats

`demcompare` results display some altitude differences **statistics**:
- the **minimal and maximal values**
- the **mean and the standard deviation**
- the **median and the NMAD** (see the References section)
- the **'90 percentile'**: 90% percentile of the absolute differences between errors and the mean (errors)
- the **percentage of valid points** (for whom the statistics have been computed)

Those stats are actually displayed by stats sets (see next section).

#### The stats sets

`demcompare` can set how the stats shall be classified to look at particular areas compared together with `to_be_classification_layers`
and / or the `classification_layers` parameters in the `stats_opts` stats section.

##### Slope default classification
The default behavior is to use the **`slope`** of both DEMs to classify the stats by slope range. The slope range can be set with the `ranges` argument. \
The statistics will then be displayed overall and by stats set.
Each set will contain all the pixels for whom the inputRef slope is contained inside the associated slope range. \
At the end, there is a stats set for each slope range and all the stats set form a partition of the altitude differences image in `final_dh.tif` file.

##### Other classifications

It is also possible not to classify the stats by slope range but to use instead any other exogenous data. \
 For that purpose, `to_be_classification_layers` and / or the `classification_layers` parameters have to be used:
- `to_be_classification_layers` are layers (exogenous rasters) that could be use as classification layers by the use of a `ranges` list. Hence, the slope layer `demcompare` computes itself belongs to this category.
- `classification_layers` are layers (exogenous raster) such as segmentation or semantic segmentation for which pixels are gathered inside superpixels whose values are shared by every pixels in a superpixel and called a label value.

For every exogenous layer (example above `land_cover`), the user ought to specify each superimposable DEM. `ref` and `dsm` keywords are then  designed to register the path of the exogenous layer, respectively superimposable to the `ref` or the `dsm`.

The user can set as many exogenous layers to classify the stats from: land cover map, validity masks, etc.
All of them will be used separately to classify the stats, and then be merged into a full classification layer that will also be used to classify the stats
(in that case `demcompare` could display the results for 'elevated roads' for which pixels are 'valid pixels').

#### The cross classification and the modes

Along with classifying the statistics, `demcompare` can display the stats in three different modes. A **mode** is
a set of all the pixels of the altitude differences image.

As shown in previous section, `demcompare` will classify stats according to classification layers (computed slopes or exogenous data provided by the user).
For each classification layer, `demcompare` knows if it is superimposable to the `ref` or the `dsm` to be evaluated. Now one could provides two land cover classification layers to `demcompare`.
One that would come with the `ref` DEM. And one that would come with the `dsm`. In this case, `demcompare` provides a three modes stats display.

Now here is how the modes are defined:
1. the **standard mode** results on all valid pixels.
  - This means nan values but also ouliers (if `remove_outliers` was set to True) and masked ones are discarded. Note that the nan values can be originated from the altitude differences image and / or the exogenous classification layers themselves.

2. the **coherent mode** is the standard mode where only the pixels sharing the same label for both DEMs classification layers are kept.
  - Say after a coregistration, a pixel P is associated to a 'grass land' inside a `ref` classification layer named `land_cover` and a `road` inside the `dsm` classification layer also named `land_cover`, then P is not coherent for `demcompare`.

3. the **incoherent mode** which is the coherent one complementary.

#### The elevation thresholds (Experimental)

The tool allows to configure `elevation_thresholds` parameter with a list of thresholds.\
For each threshold,  `demcompare` will
compute the ratio  of pixels for which the altitude difference is larger than this particular threshold.

Note:  So far results are only visible inside `stats_results-*.json` output files (see next chapter). Please also
note that the threshold is compared against the altitude differences being signed. This means that the result is not
always relevant and this stats computation shall be used carefully.

#### The dh map and the intermediate data

`demcompare` will store several data and here is a brief explanation for each one.

First, the images :

- `intial_dh.tif` it the altitude differences image when both DEMs have been reprojected to the same **inputDSM** grid and no coregistration has been performed.

- `final_dh.tif` is the altitude differences image from the reprojected DEMs after the coregistration.

- the `dh_col_wise_wave_detection.tif` and `dh_row_wise_wave_detection.tif` are respectively computed by substituting
the `final_dh.tif` average col (row) to `final_dh.tif` itself. It helps to detect any residual oscillation.


- the `coreg_DSM.tif` and `coreg_Ref.tif` are the coregistered DEMS.

- the images whose names start with 'AltiErrors-' are the plots saved by `demcompare`. They show histograms by stats
set and the same histograms fitted by gaussian.

Then, the remaining files :

- the `final_config.json` is the configuration used for a particular demcompare run.
  - It is the completion of the initial `config.json` file given by the user and contains additional information.  It can be used to relaunch the same `demcompare` run or for a step by step run.

- the files beginning with **'stats_results-'** are the `.json` and `.csv` files containing the statistics for each set. There is one file by mode.

- the `histograms/*.npy` files are the numpy histograms for each stats mode and set.

#### Generated output documentation

The output `<test_output>/doc/published_report/` directory contains a full generated [sphinx](https://www.sphinx-doc.org/) documentation with all the results presented
for each mode and each set, in `html` or `latex` format.

It can be regenerated using `make html` or `make latexpdf` in `<test_output>/doc/src/` directory

## Dependencies

The full list of dependencies can be observed from the [setup.cfg](./setup.cfg) file.

## Licensing

demcompare software is distributed under the Apache Software License (ASL) v2.0.

See [LICENSE](./LICENSE) file or http://www.apache.org/licenses/LICENSE-2.0 for details.

Copyrights and authoring can be found in [NOTICE](./NOTICE) file.

## References

For more details about the NMAD metric : \
Höhle, J., Höhle, M., 2009. *Accuracy assessment of Digital Elevation Models by means of robust statistical methods.*
 ISPRS Journal of Photogrammetry and Remote Sensing 64(4), 398-406.

For the Nuth & Kääb universal coregistration algorithm :\
Nuth, C. Kääb, 2011. *A. Co-registration and bias corrections of satellite elevation data sets for quantifying glacier
thickness change.* Cryosphere 5, 271290.
