.. _DEM_processing_methods:

DEM processing methods
======================

**Demcompare** can compute a wide variety of **statistics** on either an input DEM, or the difference between two input DEMs.

Before the **statistics** computation, the input DEM or the difference between two input DEMs is processed by a **DEM processing method**.

Several DEM processing methods are available.

The DEM processing methods can be split in two categories: DEM processing methods applied on one DEM, and DEM processing methods applied on two DEMs:

.. _list_DEM_processing_methods:

.. tabs::

  .. tab:: Applied on one DEM 

    - ``ref``: returns the reference DEM
    - ``sec``: returns the secondary DEM
    - ``ref-curvature``: computes and returns the curvature of the reference DEM
    - ``sec-curvature``: computes and returns the curvature of the secondary DEM

  .. tab:: Applied on two DEMs
      
    - ``alti-diff``: computes and returns the difference in altitude between the two input DEMs
    - ``alti-diff-slope-norm``: computes and returns the difference in altitude between the two input DEMs, and normalizes it by the slope
    - ``angular-diff``: computes and returns the angular difference between the two input DEMs

.. warning::
    For ``sec`` and ``sec-curvature`` DEM processing methods, the classification layers are taken from the reference DEM.

.. note::

   More information about the curvature, the difference in altitude between the two input DEMs normalized by the slope and the angular difference can be found in :ref:`curvature`, :ref:`slope_normalized_elevation_difference` and :ref:`angular_difference` respectively.

After the DEM processing methods, statistics can be computed on the resulting DEM.

Stats computation
*****************

In summary, the workflow can be represented by the figure below:

.. figure:: /images/workflow.png
            :width: 700px
            :align: center

All the different DEM processing methods can be used within a single configuration file as below (a **coregistration** can be added):

.. code-block:: json

        {
            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "nodata": -9999.0
            },
            "input_sec": {
                "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
                "nodata": -32768
            }
            "statistics": {
                "alti-diff": {
                    "remove_outliers": true
                },
                "alti-diff-slope-norm": {
                    "remove_outliers": true
                },
                "angular-diff": {
                    "remove_outliers": true
                },
                "ref": {
                    "remove_outliers": true
                },
                "sec": {
                    "remove_outliers": true
                },
                "ref-curvature": {
                    "remove_outliers": true
                },
                "sec-curvature": {
                    "remove_outliers": true
                }
            }
        }

By default, the following metrics will be computed: ``mean``, ``median``, ``max``, ``min``, ``sum``, ``squared_sum``, ``std``, ``percentil_90``, ``nmad``, ``rmse``, ``pdf``, ``cdf``, ``hillshade``, ``svf``.

The user may specify the required metrics as follows:

.. code-block:: json

        {
            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "nodata": -9999.0
            },
            "input_sec": {
                "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
                "nodata": -32768
            }
            "statistics": {
                "alti-diff": {
                    "remove_outliers": true,
                    "metrics": ["mean", {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}}]
                }
            }
        }

The DEM processing methods applied on one DEM can also be used with a single DEM as input, as below:

.. code-block:: json

        {
            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "nodata": -9999.0
            },
            "statistics": {
                "ref": {
                    "remove_outliers": true
                },
                "ref-curvature": {
                    "remove_outliers": true
                }
            }
        }

By default, the following metrics will be computed: ``mean``, ``median``, ``max``, ``min``, ``sum``, ``squared_sum``, ``std``.

The user may specify the required metrics as follows:

.. code-block:: json

        {
            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "nodata": -9999.0
            },
            "statistics": {
                "ref": {
                    "remove_outliers": true,
                    "metrics": ["mean", {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}}]
                }
            }
        }

See also:

.. toctree::
   :maxdepth: 4

   angular_difference.rst
   slope_normalized_elevation_difference.rst