.. _input_DEM:

Input DEM
=========

Basic input DEMs configuration
******************************

A possible input DEM configuration would be the following:

.. code-block:: json

      "input_ref": {
        "path": "./Gironde.tif",
      },
      "input_sec": {
        "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
      }


Geoid reference
***************

**Demcompare** requires the input DEMs to be geotiff files **.tif**.

The DEMs altitudes can rely on both ellipsoid and geoid references, **being the ellipsoid the reference by default** if not
indicated otherwise. If DEMs altitudes are to rely on **geoid**, the configuration should be:

.. code-block:: json

    "input_sec" : {  "path":"./input_sec.tif"
                     "geoid_georef" : "True",
                  }

In this case, **EGM96 geoid** will be used by default.

Otherwise, the absolute path to a locally available geoid model can be given. The geoid local model should be either a *GTX*, *NRCAN* or *NTv2* file.

For instance, if DEMs altitudes are to rely on a local *.gtx* available **geoid** model, the configuration should be:

.. code-block:: json

    "input_sec" : {  "path": "./input_sec.tif"
                     "geoid_georef": "True",
                     "geoid_path": "path/to/egm08_25.gtx"
                   }


ROI
***

To limit DEM comparison to a Region Of Interest (ROI) one can set a bouding box in terrain geometry or part of the DEM with image coordinates:


.. tabs::

  .. tab:: ROI with Terrain Coordinates
  
    The geographical coordinates of the image defines as tuple with *(left, bottom, right, top)* coordinates. For instance, for a DSM whose Coordinate Reference System is **EPSG:32630**, a possible ROI would be *(left=600255.0, bottom=4990745.0, right=709255.0, top=5099745.0)*.
    
    .. code-block:: json 

        "input_ref": {
          "path": "./Gironde.tif",
        },
        "input_sec": {
          "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
          "roi": {
                "left": 40.5,
                "bottom": 38.0,
                "right": 44.0,
                "top": 41.0
              }
        }

  .. tab:: ROI with Image Coordinates

    The image coordinates *(x,y)* of its top-left corner and its dimensions (w, h) in pixels defines as a python dictionary with `x`, `y`, `w` and `h` keys.

    .. code-block:: json

      "input_ref": {
          "path": "./Gironde.tif",
        },
        "input_sec": {
          "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
          "roi": {
                "x": 50,
                "y": 100,
                "w": 1000, 
                "h": 500
              }
        }

Altimetric unit
***************

Because it can happen that both DEMs would not have been produced with the same altimetric unit, the ``zunit`` parameter might be useful at times. 
It allows one to explicitly provide both DEMs unit, so that demcompare can convert z values adequately. The default ``zunit`` value is ``m``. 

.. code-block:: json

    "input_ref": {
        "path": "./Gironde.tif",
        "zunit": "cm",

      },
      "input_sec": {
        "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
        "zunit": "m",
      }


Nodata
******

Demcompare will try to read the nodata value of each DEM from their metadata. However, if for some reasons another nodata value shall be specified then one can use the `nodata` parameter.

.. code-block:: json

    "input_ref": {
        "path": "./Gironde.tif",
        "nodata": -9999.0,

      },
      "input_sec": {
        "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
        "nodata": -32768,
      }

Input DEMs parameters
*********************

Here is the exhaustive list of parameters one can use for the input DEMs. Along with the parameters are the associated default values (when relevant).
Every parameter here is a key for either the ``input_ref`` or the ```input_sec`` root parameter.

.. csv-table:: Input DEMs parameters
  :header: "Name", "Description", "Type", "Default value", "Required"
  :widths: auto
  :align: left

  ``'path'``, "Path", "string", ``None``, "Yes"
  ``'roi'``, "Processed Region of interest of the input Sec", "Dict", ``None``, "No"
  ``'geoid_georef'``, "True if the georef of the input Ref", "string", ``False``, "No"
  ``'geoid_path'``, "Geoid path of the input Ref", "string", ``None``, "No"
  ``'zunit'``, "Z axes unit", "string", ``m``, "No"
  ``''nodata'``, "No data value of the input Ref", "int", ``None``, "No"
  ``'classification_layers':{'name_map_path':}``, "Path to the classification layer map", "string", ``None``, "No"

.. note::

  ``'classification_layers':{'name_map_path':}`` is a parameter used for statistics purpose. See :ref:`statistics` for more information.

Be aware that for a command line execution, one must set the directory where data should be written down.

.. csv-table::
    :header: "Name","Description", "Type", "Default value", "Required"
    :widths: auto
    :align: left

    ``'output_dir'``,Output directory path,string, ``None``, Oui

.. note::

  Demcompare accepts a single DEM as input. If it is the case, it must be defined as the ``input_ref``.

