.. _statistics:


Statistics
==========

.. note::

    In this chapter we use *ref* and *sec* abbreviations when refering to the reference input DEM (``input_ref``) and the secondary input DEM (``ìnput_sec``) respectively.


**Demcompare** can compute a wide variety of **statistics** on either an input DEM, or the difference between two input DEMs.
The statistics module can consider different number of inputs:

.. tabs::

    .. tab:: One input dem

        .. code-block:: json

            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "nodata": -9999.0,
            },
            "statistics": {
                "remove_outliers": "True",
            }

        If **one single DEM** is specified in the configuration, the input or default metrics will be directly computed on the input DEM.

        .. figure:: /images/stats_input_one_dem.png
            :width: 300px
            :align: center

            Statistics computation with one input DEM.



        By default, the following metrics will be computed:  ``mean``, ``median``, ``max``, ``min``, ``sum``, ``squared_sum``, ``std``.

        The user may specify the required metrics as follows:


        .. code-block:: json

            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "nodata": -9999.0,
            },
            "statistics": {
                "remove_outliers": "True",
                "metrics": ["mean", {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}}]
            }

    .. tab:: Two input dems

        .. code-block:: json

            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "nodata": -9999.0,
            },
            "input_sec": {
                "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
                "nodata": -32768,
            }
            "statistics": {
                "remove_outliers": "True",
            }

        If **two DEMs** are specified in the configuration, demcompare will do the reprojection of both DEMs to have the same resolution
        and size, and the difference between both reprojected DEMs will be considered to compute the input or default metrics.

        .. figure:: /images/stats_input_two_dems.png
            :width: 800px
            :align: center

            Statistics computation with two input DEMs.

        By default, the following metrics will be computed:  ``mean``, ``median``, ``max``, ``min``, ``sum``, ``squared_sum``, ``std``,
        ``percentil_90``, ``nmad``, ``rmse``.

        The user may specify the required metrics as follows:

        .. code-block:: json

            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "nodata": -9999.0,
            },
            "input_sec": {
                "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
                "nodata": -32768,
            }
            "statistics": {
                "remove_outliers": "True",
                "metrics": ["mean", {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}}]
            }


With the coregistration step
*****************************

If both coregistration and statistics steps are present on the input configuration:

- In order to evaluate the coregistration effect, the differences between the reprojected DEMs before and after coregistration, named **initial_dem_diff** and **final_dem_diff**, will be considered to compute the Probability Density Function and the Cumulative Density Function.
- The difference between the reprojected DEMs after coregistration (the **final_dem_diff**) will be considered to compute the input or default metrics.


.. code-block:: json

    "output_dir": "./test_output/",
    "input_ref": {
        "path": "./Gironde.tif",
        "nodata": -9999.0,
    },
    "input_sec": {
        "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
        "nodata": -32768,
    },
    "coregistration": {
        "coregistration_method": "nuth_kaab_internal",
    }
    "statistics": {
        "remove_outliers": "True",
    }

.. figure:: /images/stats_input_after_coreg.png
    :width: 800px
    :align: center

    Statistics computation after the coregistration step.

The following metrics will be computed:


.. tabs::

  .. tab:: Metrics **to evaluate the coregistration effect**
      On **initial_dem_diff** and on **final_dem_diff**: ``cdf``, ``pdf``.

    .. note::
        No classification is considered for the metrics to evaluate the coregistration effect.
        If classification layers are specified on the input configuration, those will be only be considered for the
        ''Other default metrics'' computation.

  .. tab:: Default metrics
      On **final_dem_diff**: ``mean``, ``median``, ``max``, ``min``, ``sum``, ``squared_sum``, ``std``, ``percentil_90``, ``nmad``, ``rmse``.

    .. note::
        If the user specifies the required metrics to be computed, those will substitute the default metrics. However,
        the ''metrics to evaluate the coregistration effect'' will still be computed.


The user may specify the required metrics as follows :

.. code-block:: json

    "output_dir": "./test_output/",
    "input_ref": {
        "path": "./Gironde.tif",
        "nodata": -9999.0,
    },
    "input_sec": {
        "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
        "nodata": -32768,
    },
    "coregistration": {
        "coregistration_method": "nuth_kaab_internal",
    }
    "statistics": {
        "remove_outliers": "True",
        "metrics": ["mean", {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}}]
    }



Metrics
*******

The following metrics are currently available on demcompare:

.. tabs::

  .. tab:: Scalar metrics

    - ``mean``
    - ``max``
    - ``min``
    - ``std`` (Standard Deviation)
    - ``rmse`` (Root Mean Squared Error)
    - ``median``
    - ``nmad`` (Normalized Median Absolute Deviation) = :math:`1.486*median(\lvert data - median(data)\rvert)`
    - ``sum``
    - ``squared_sum``
    - ``percentil_90``

  .. tab:: Vector metrics
      .. csv-table::
        :header: "Name", "Type", "Parameters", "Type", "Default value"
        :widths: auto
        :align: left

          ``'cdf'``\ Cumulative Density Function,vector,bin_step, "float", ``0.1``
          ,,output_csv_path, "string",``None``
          ,,output_plot_path, "string",``None``
          ``'pdf'``\ Probability Density Function,vector,bin_step, "float",``0.2``
          ,,width, "float",``0.7``
          ,,filter_p98, "float",``"False"``
          ,,output_csv_path, "string",``None``
          ,,output_plot_path, "string",``None``
          ``'ratio_above_threshold'``,vector,elevation_threshold, "List[float, int]", ":math:`[0.5, 1, 3]`"
          ,,original_unit, "string",``"m"``
          ,,output_csv_path, "string",``None``

.. note::

    The metrics are always computed on **valid pixels**. Valid pixels are those whose value is different than NaN and the
    nodata value (-32768 by default if not specified in the input configuration or in the input DEM).

.. note::
    Apart from only considering the valid pixels, the user may also specify the ``remove_outliers`` option
    in the input configuration. This option will also **filter all DEM pixels outside (mu + 3 sigma) and (mu - 3 sigma)**,
    being *mu* the *mean* and *sigma* the *standard deviation* of all valid pixels in the DEM.

Classification layers
*********************

Classification layers are a way to classify the DEM pixels in classes according to different criteria in order to compute specific statistics according to each class.

Four types of classification layers exist:

.. tabs::

    .. tab:: Global 

        The global classification is the default classification and is **always computed**.
        This layer has a single class where all valid pixels are considered. If no classification layers are specified in the input configuration,
        only the global classification will be considered.

    .. tab:: Segmentation 


        This type of classification layer considers an **input classification mask** in order to classify the DEM pixels.
        The classification mask must be specified with its classes, and linked to one of the input DEMs defined in the input configuration as follows:

        .. code-block:: json

            "output_dir": "./test_output/",
            "input_ref": {
                "path": "./Gironde.tif",
                "zunit": "m"
            },
            "input_sec": {
                "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
                "zunit": "m",
                "nodata": -9999,
                "classification_layers": {
                    "Status": {
                        "map_path": "./FinalWaveBathymetry_T30TXR_20200622T105631_Status.TIF"
                    }
                }
            }
            "statistics": {
                "remove_outliers": "False",
                "classification_layers": {
                    "Status": {
                        "type": "segmentation",
                        "classes": {"valid": [0],"KO": [1],"Land": [2],"NoData": [3],"Outside_detector": [4]}
                    }
                }
            }

        On this example, we can see that the classification mask is linked to the secondary DEM.

        Regarding the classification_layer configuration,
        the ``type`` is specified as ``segmentation``, and the different ``classes`` are specified as a dictionary containing the different
        names and their mask values.

        Notice that a class may contain different mask values, for instance:

        .. code-block:: json

                "statistics": {
                    "remove_outliers": "False",
                    "classification_layers": {
                        "Status": {
                            "type": "segmentation",
                            "classes": {"valid": [0, 1], "Land": [2, 3], "NoData": [4, 5]}
                        }
                    }
                }

        If a classification mask is specified for both *input_ref* and *input_sec*, the mask classification of the **ref** DEM is considered
        for the **general** statistics computation, whilst the **sec** mask classification is considered for the **intersection** and **exclusion** statistics as
        explained on :ref:`modes`.


        .. note::
            The input classification mask must be **superimposable** to its support DEM, meaning that it must have the **same size and resolution**.
            It is to be noticed that during execution, all the transformations applied to the support DEM will also be applied to its classification
            masks to ensure that they continue to be superimposable.

    .. tab:: Slope 


        This type of classification **computes the slope** of the input DEMs and classifies the pixels according to the **range** on which its slope falls.
        It is to be noticed that if two DEMs are defined as inputs, then the slope will be computed on both input DEMs **separately**, and not in the difference between both.

        The slope of each DEM is obtained as follows:

            .. math::

                Slope_{DEM}(x,y) &= \sqrt{(gx / res_x)^2 + (gy / res_y)^2)} / 8


            , where :math:`c_{gx}` and :math:`c_{gy}` are the result of the convolution :math:`c_{gx}=conv(DEM,kernel_x)` and :math:`c_{gy} = conv(DEM,kernel_y)` of the DEM with the kernels :


            .. math::

                kernel_x = \begin{bmatrix}-1 & 0 & 1\\-2 & 0 & 2\\-1 & 0 & 1\end{bmatrix}


            .. math::
                kernel_y = T(kernel_x)


        The slope will then be classified by the **ranges** set with the ``ranges`` argument.

        Each class will contain all the pixels for whom the slope is contained inside the associated slope range. At the end, there will be a class mask for each slope range.

        Regarding the classification_layer configuration,
        the ``type`` is specified as ``slope``, and the different ``ranges`` are specified as a list. A valid **slope** configuration could be:

        .. code-block:: json

            "classification_layers": {
                "Slope0": {
                    "type": "slope",
                    "ranges": [0, 5, 10, 25, 45]
                }
            }



    .. tab:: Fusion 



        This type of classification layer is created from two or more existing classification layers,
        as it is the result of **fusing the classes of different classification layers**.
        It is to be noticed that **only classification layers belonging to the same support DEM can be fused**.

        For example, given the two following classification layers with their corresponding classes and mask values:

        .. code-block:: bash

            Slope0: "[0%;5%[", 1
                    "[5%;10%[", 2
                    "[10%;inf[", 3
            Status: "Sea", 1
                    "Deep_land", 2
                    "Coast", 3

        The resulting fusion layer would have the following fused classes :

        .. code-block:: bash

            Fusion0: "Status_sea_&_Slope0_[0%;5%[", 1,
                        "Status_sea_&_Slope0_[5%;10%[", 2,
                        "Status_sea_&_Slope0_[10%;inf[", 3,
                        "Status_deep_land_&_Slope0_[0%;5%[", 4,
                        "Status_deep_land_&_Slope0_[5%;10%[", 5,
                        "Status_deep_land_&_Slope0_[10%;inf[", 6,



        A possible configuration including a fusion classification layer in included here. As one can see the ``type`` is specified as ``fusion``,
        and the support dem of the list of layers to be fused, in this case ``sec``, must be specified :


        .. code-block:: json

                "output_dir": "./test_output/",
                "input_ref": {
                    "path": "./Gironde.tif",
                    "zunit": "m"
                },
                "input_sec": {
                    "path": "./FinalWaveBathymetry_T30TXR_20200622T105631_D_MSL_invert.TIF",
                    "zunit": "m",
                    "nodata": -9999,
                    "classification_layers": {
                        "Status": {
                            "map_path": "./FinalWaveBathymetry_T30TXR_20200622T105631_Status.TIF"}
                    }
                },
                "statistics": {
                    "classification_layers": {
                        "Status": {
                            "type": "segmentation",
                            "classes": {"valid": [0], "KO": [1], "Land": [2], "NoData": [3], "Outside_detector": [4],
                        },
                        "Slope0": {
                            "type": "slope",
                            "ranges": [0, 10, 25, 50, 90],
                        },
                        "Fusion0": {
                            "type": "fusion",
                            "sec": ["Slope0", "Status"]
                        }
                    }
                }

        In the following schema we can see an example case where two different segmentation layers and a slope layer
        are created, each having a single support:

            - Segmentation_0 has **ref** support
            - Segmentation_1 has **sec** support
            - Slope_0 has **sec** support

        Hence, a **fusion layer** can be created by **fusing the two layers that have the same support, in this case Segmentation_1**
        **and Slope_0 with sec support**.


        .. figure:: /images/stats_fusion_schema.png
            :width: 750px
            :align: center

            Statistics schema with a fusion layer.


.. _modes:

The modes
~~~~~~~~~


    As shown in previous section, **demcompare** will classify stats according to classification layers and classification layer masks must be superimposable to one DEM, meaning that the classification mask and its support DEM must have the same size and resolution.
    
    Whenever a classification layer is given for both DEMs (say one has two DEMs with associated segmentation maps) then it can be possible to observe the metrics for pixels whose classification (segmentation for example) is the same between both DEM or not.
    These observations are available through what we call `mode`. Demcompare supports:


.. tabs::

  .. tab:: The **standard mode**

       Within this mode **all valid pixels are considered**. It means nan values but also outliers (if ``remove_outliers`` was set to ``"True"``) and masked ones are discarded.

       Note that the nan values can be originated from the altitude differences image and / or the exogenous classification layers themselves (ie. if the input segmentation
       has NaN values, the corresponding pixels will not be considered for the statistics computation of this classification layer).

  .. tab:: The **intersection** and **exclusion** modes
       These modes are only available if both DEMs (**ref** and **sec**) where classified by the same classification layer :

       The **intersection mode** is the mode where **only the pixels sharing the same label for both DEMs classification layers are kept**.

        - Say after a coregistration, a pixel *P* is associated to a 'grass land' inside a `ref` classification layer named `land_cover` and a `road` inside the `sec` classification layer also named `land_cover`, **then pixel P is not intersection** for demcompare.

       The **exclusion mode** which is the intersection one complementary.

In the following schema we can see a scenario where two different segmentation layers and a slope layer
are created. Both segmentation layers having a single support and the slope layer having **two supports**.

- Segmentation_0 has **only ref** support, hence the statistics are computed considering the **ref** segmentation_0_mask.
- Segmentation_1 has **only sec** support, hence the statistics are computed considering the **sec** segmentation_1_mask.
- Slope_0 has both ref and support, hence the statistics are computed considering:

    - the **ref** slope_0_mask for the **standard** mode
    - the intersection between the **ref** slope_0_mask and the **sec** slope_0_mask for the **intersection** and **exclusion** modes.

.. figure:: /images/stats_support_schema.png
    :width: 750px
    :align: center

    Statistics schema with intersection and exclusion modes.

Metric selection
~~~~~~~~~~~~~~~~


    The metrics to be computed **may be specified at different levels** on the statistics configuration:

     - **Global level**: those metrics will be computed for all classification layers
     - **Classification layer level**: those metrics will be computed specifically for the given classification layer

    For instance, with the following configuration we could compute the *mean, ratio_above_threshold* metrics on **all layers**, whilst
    *nmad* metric would be computed **only for the Slope0 layer**.

    .. code-block:: json

          "statistics": {
            "classification_layers": {
                "Status": {
                    "type": "segmentation",
                    "classes": {
                        "valid": [0],
                        "KO": [1],
                        "Land": [2],
                        "NoData": [3],
                        "Outside_detector": [4],
                    },
                },
                "Slope0": {
                    "type": "slope",
                    "ranges": [0, 10, 25, 50, 90],
                    "metrics": ["nmad"],
                },
                "Fusion0": {
                    "type": "fusion",
                    "sec": ["Slope0", "Status"]
                },
            },
            "metrics": [
                "mean",
                {"ratio_above_threshold": {"elevation_threshold": [1, 2, 3]}},
            ],
           }


Statistics parameters
*********************

.. tabs::

  .. tab:: statistics

    Here is the list of the parameters of the input configuration file for the statistics step and its associated default value when it exists:

    +---------------------------------------------+-------------------------------------------------+-------------+------------------------------+----------+
    | Name                                        | Description                                     | Type        | Default value                | Required |
    +=============================================+=================================================+=============+==============================+==========+
    | ``remove_outliers``                         | | Remove outliers during statistics             | string      | ``"False"``                  | No       |
    |                                             | | computation                                   |             |                              |          |
    +---------------------------------------------+-------------------------------------------------+-------------+------------------------------+----------+
    | ``metrics``                                 | Metrics to be computed                          | List        | ``List of default metrics``  | No       |
    |                                             |                                                 |             |                              |          |
    +---------------------------------------------+-------------------------------------------------+-------------+------------------------------+----------+

  .. tabs::
    .. tab:: classification_layers

        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+
        | Name                       | Description                                     | Type        | Default value                       | Required |
        +============================+=================================================+=============+=====================================+==========+
        | ``type``                   | Classification layer type                       | string      |  ``None``                           | Yes      |
        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+
        | | ``remove_outliers``      | | Remove outliers during statistics computation | string      | ``Value set for the whole stats``   | No       |
        | |                          | | for this particular classification layer      |             |                                     |          |
        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+
        | ``nodata``                 | Classification layer no data value              | float or int|  ``-32768``                         | No       |
        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+
        | ``metrics``                | | Classification layer metrics to be computed   | List        | ``List of default metrics``         | No       |
        |                            | | (if metrics have been specified for the whole |             |                                     |          |
        |                            | | stats, they will also be computed for this    |             |                                     |          |
        |                            | | classification)                               |             |                                     |          |
        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+

    .. tabs::
      .. tab:: Segmentation type

        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+
        | Name                       | Description                                     | Type        | Default value                       | Required |
        +============================+=================================================+=============+=====================================+==========+
        | ``'classes'``              | Segmentation classes                            | Dict        |  ``None``                           | Yes      |
        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+

      .. tab:: Slope type

        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+
        | Name                       | Description                                     | Type        | Default value                       | Required |
        +============================+=================================================+=============+=====================================+==========+
        | ``'ranges'``               | Slope ranges                                    | List        |  ":math:`[0, 5, 10, 25, 45]`"       | No       |
        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+

      .. tab:: Fusion type

        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+
        | Name                       | Description                                     | Type        | Default value                       | Required |
        +============================+=================================================+=============+=====================================+==========+
        | ``'ref'``                  | Ref classification layers to fusion             | List        | ``None``                            | No       |
        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+
        | ``'sec'``                  | Sec classification layers to fusion             | List        | ``None``                            | No       |
        +----------------------------+-------------------------------------------------+-------------+-------------------------------------+----------+


Statistics outputs
******************

Output files and their required parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The images and files saved with the ``statistics`` option activated on the configuration :


+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| Name                                                               | Description                                                                              |
+====================================================================+==========================================================================================+
| *dem_for_stats.tif*                                                | DEM on which the statistics have been computed                                           |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| *ref and sec_rectified_support_map.tif*                            | | Stored on each classification layer folder, the rectified support maps                 |
|                                                                    | | where each pixel has a class value.                                                    |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| *stats_results.csv and .json*                                      | | Stored on each classification layer folder,                                            |
|                                                                    | | the CSV and Json files storing the computed statistics by class.                       |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| *stats_results_intersection.csv and .json*                         | | Stored on each classification layer folder, the CSV and Json files                     |
|                                                                    | | storing the computed statistics by class in mode intersection.                         |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+
| *stats_results_exclusion.csv and .json*                            | | Stored on each classification layer folder, the CSV and Json files                     |
|                                                                    | | storing the computed statistics by class in mode exclusion.                            |
+--------------------------------------------------------------------+------------------------------------------------------------------------------------------+

Output directories
~~~~~~~~~~~~~~~~~~

With the command line execution, the following statistics directories that may store the respective files will be automatically generated.


.. code-block:: bash

    .output_dir
    +-- stats
        +-- dem_for_stats.tif
        +-- *classification_layer_name*
            +-- stats_results.json/csv
            +-- stats_results_intersection.json/csv
            +-- stats_results_exclusion.json/csv
            +-- ref_rectified_support_map.tif
            +-- sec_rectified_support_map.tif

.. note::
    Please notice that even if no classification layer has been specified, the results will be stored in a folder called ``global``, as it
    is the classification layer that is always computed and only considers all valid pixels.

.. note::
    Please notice that some data may be missing if it has not been computed for the classification layer (ie. intersection maps are only computed under certain conditions :ref:`modes`).
