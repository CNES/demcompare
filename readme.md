# dem_compare

This python software aims at comparing two DSMs together, whether or not they share common format, projection system,
planimetric resolution, and altimetric unit.

The coregistration algorithm is based on the Nuth & Kääb universal coregistration method.

dem_compare provides a wide variety of standard metrics and allows one to classify the statistics. The default behavior
classifies the stats by slope ranges but one can provide any other data to classify the stats from.

A comparison report can be compiled as html or pdf documentation with statistics printed as tables and plots.

## Usage

Run the python script `dem_compare.py` with a json configuration file as unique
argument:

    python dem_compare.py example_config.json

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

The processed Region of interest (ROI) is either defined by:

    the image coordinates (x,y) of its top-left corner, and its dimensions (w, h) in pixels as a python dictionary with
    'x', 'y', 'w' and 'h' keys
    or the geographical coordinates of the projected image as tuple with (left, right, bottom, top) coordinates

In anyway, this is four numbers that ought to be given in the `json` configuration file.

If no ROI definition is provided then DEMs raster are fully processed.

## Dependencies

Here is the list of required dependencies for the python environment:

    `gdal` with version 2.1.0 or higher.
    `numpy`
    `scipy`
    `pyproj`
    `astropy`
    `matplotlib`

For the report to be compied one shall install `latex` and `latex` (for the .pdf version).

## References

For more details about the NMAD metric :
Höhle, J., Höhle, M., 2009. *Accuracy assessment of Digital Elevation Models by means of robust statistical methods.*
 ISPRS Journal of Photogrammetry and Remote Sensing 64(4), 398-406.

For the Nuth & Kääb universal coregistration algorithm :
Nuth, C. Kääb, 2011. *A. Co-registration and bias corrections of satellite elevation data sets for quantifying glacier
thickness change.* Cryosphere 5, 271290.
