.. _coregistration:

Coregistration
==============

.. note::

    In this chapter we use *ref* and *sec* abbreviations when refering to the reference input DEM (``input_ref``) and the secondary input DEM (``ìnput_sec``) respectively.


Introduction
************

A possible coregistration configuration would be the following:

.. code-block:: json

    "coregistration": {
        "method_name": "nuth_kaab_internal"
    }

Be aware that **demcompare**'s coregistration is performed in three steps:

i) Reprojection
---------------

Because radiometric differences are used to estimate the planimetric and altimetric offsets between both DEMs, **demcompare** needs both DEMs to rely on the same geographic grid.
This means that operations such as crop and resampling are first performed to ensure both DEMs can actually be compared pixelwise:

- the crop is done based on the DEMs a priori intersection 
- the resampling ensures both DEMs share the same resolution. The default resolution is the one of the ``input_sec`` DEM. 

The output of this step are two DEMs intuitively called ``reproj_REF.tif`` and ``reproj_SEC.tif``.

ii) Offsets estimation
----------------------

This is the actual coregistration part. Demcompare will use one implemetation of [NuthKaab]_ algorithm to estimate the ``x``, ``y``, and ``z`` shifts between reprojected DEMs (namely ``reproj_REF`` and ``reproj_SEC``).

The output of this step are two DEMs intuitively called ``reproj_coreg_REF.tif`` and ``reproj_coreg_SEC.tif``.


iii) Shift of the ``input_sec`` DEM from ``x`` and ``y`` planimetric offsets
----------------------------------------------------------------------------

The last step consists in changing the origin of the ``input_sec`` DEM so that it now matches the ``input_ref`` ones. 

The output of this step is then a ``coreg_SEC.tif`` DEM. One can now open the ``input_ref``  and the ``coreg_SEC.tif`` in a GIS viewer and, hopefully, observe no residual shift. 


Schematic overview
------------------

.. figure:: /images/schema_coregistration.png
    :width: 1000px
    :align: center

    Demcompare's coregistration schema

.. note:: Please notice that both *reproj_coreg_ref* and *reproj_coreg_sec* share the same georeference origin, **but this origin may not be the same as the origin of reference DEM**. Hence, they shall only be used for computing altitude difference for statistical purposes.

.. note:: Notice that if a single DEM is given as input, the coregistration step cannot be computed obviously.

.. warning::
  Be careful that the coregistration altimetric ``z`` shift is given as output information but is not used for dem coregistration in demcompare. The altimetric shift can be from many sources and its correction could blur the comparison analysis.


Detailed parameters
*******************

Sampling source
---------------

By default, both reprojected DEMs will have **sec**'s resolution. However, one may consider **ref**'s resolution specifying the coregistration's **sampling_source** parameter
on the input coregistration configuration.

A possible coregistration configuration with reference's resolution would be the following:

.. code-block:: json

    "coregistration": {
        "method_name": "nuth_kaab_internal",
        "sampling_source": "ref"
    }


Initial shift
-------------

The user may have a **prior estimation** of the shift between the input DEMs. In this case, the parameters
`estimated_initial_shift_x` and `estimated_initial_shift_y` may be specified.
If the estimated initial shifts are given, demcompare will apply them to the *input_sec* DEM before the coregistration algorithm.

A possible coregistration configuration would be the following:

.. code-block:: json

    "coregistration": {
        "method_name": "nuth_kaab_internal",
        "estimated_initial_shift_x": 2.5,
        "estimated_initial_shift_y": -0.6
    }

Number of iterations
--------------------

The number of iterations in the Nuth & Kaab algorithm can be modified, by specifying the `number_of_iterations` parameter. By default this value is set to **6 iterations**. 

A possible coregistration configuration would be the following:

.. code-block:: json

    "coregistration": {
        "method_name": "nuth_kaab_internal",
        "number_of_iterations": 10,
    }



Coregistration analysis
-----------------------

The coregistration may be analyzed by computing the **altitude difference before and after the coregistration** along with its histogram. To do so,
the user needs to specify the **statistics** step in the input configuration as follows:

.. code-block:: json

    "coregistration": {
        "method_name": "nuth_kaab_internal"
    },
    "statistics": {
    }


If the **statistics** step is specified in the input configuration file, **demcompare** will
compute the altitude differences.

.. figure:: /images/doc_ref.gif
    :width: 300px
    :align: center

    Superposition of two DSMs that need to be coregistered.

In this example, the two uncoregistered DEMs had the initial altitude difference shown on the following image.

.. figure:: /images/initial_dh.png
    :width: 260px
    :name: initial
    :align: center

    Initial altitude difference between the two DSMs.

After Nuth et Kaab coregistration, the final altitude difference between both coregistered DEMs is shown on the following image:

.. figure:: /images/final_dh.png
    :width: 260px
    :align: center

    Final altitude difference between the two coregistered DSMs.

**The altitude differences are computed with the reprojected DEMs** before (*dem_reproj_ref* and *dem_reproj_sec* on the schema) and after the coregistration
(*dem_reproj_coreg_ref* and *dem_reproj_coreg_sec* on the schema).

Full list of parameters
***********************

Scientific parameters
---------------------


.. tabs::

    .. tab:: coregistration

        Here is the list of the parameters of the input configuration file for the coregistration step and its associated default value when it exists:

        +-------------------------------+-------------------------------------------------+-------------+---------------------+----------+
        | Name                          | Description                                     | Type        | Default value       | Required |
        +===============================+=================================================+=============+=====================+==========+
        | ``method_name``               | Planimetric coregistration method               | string      | ``nuth_kaab``       | No       |
        +-------------------------------+-------------------------------------------------+-------------+---------------------+----------+
        | ``number_of_iterations``      | | Number of iterations                          | int         | ``6``               | No       |
        |                               | | of the coregistration method                  |             |                     |          |
        +-------------------------------+-------------------------------------------------+-------------+---------------------+----------+
        | ``estimated_initial_shift_x`` | | Estimated initial x                           | int         |  ``0``              | No       |
        |                               | | coregistration shift                          |             |                     |          |
        +-------------------------------+-------------------------------------------------+-------------+---------------------+----------+
        | ``estimated_initial_shift_y`` | | Estimated initial y                           | int         |  ``0``              | No       |
        |                               | | coregistration shift                          |             |                     |          |
        +-------------------------------+-------------------------------------------------+-------------+---------------------+----------+
        | ``sampling_source``           | Sampling source for reprojection                | string      | ``sec``             | No       |
        +-------------------------------+-------------------------------------------------+-------------+---------------------+----------+
        | ``save_optional_outputs``     | | If save internal DEMs and coregistration      | string      | ``"False"``         | No       |
        |                               | | method outputs such as iteration plots        |             |                     |          |
        |                               | | to disk                                       |             |                     |          |
        +-------------------------------+-------------------------------------------------+-------------+---------------------+----------+


I/O parameters
--------------

The different DEMs used and created during the coregistration step along with plots to analyze the coregistration algorithm
and performance will be saved to disk according to the input configuration.

Output files and their required parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The coregistration images and files saved to disk :

.. csv-table::
    :header: "Name","Description"
    :widths: auto
    :align: left

    ``coreg_SEC.tif``,Coregistered secondary DEM
    ``demcompare_results.json``,Output json file containing coregistration offsets
    ``logs.log``,Logging file

The images and statistics to analyze the coregistration saved with both ``coregistration`` and ``statistics`` options activated on the configuration :

+-----------------------------------------+------------------------------------------------------------------------------------------+
| Name                                    | Description                                                                              |
+=========================================+==========================================================================================+
| *initial_dem_diff.tif*                  | | Altitude differences image when both DEMs have been reprojected                        |
|                                         | | to the same grid and no coregistration has been performed                              |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *final_dem_diff.tif*                    | | Altitude differences image from the reprojected DEMs after                             |
|                                         | | the coregistration.                                                                    |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *initial_dem_diff_snapshot.png*         | Snapshot plot of `initial_dem_diff.tif`                                                  |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *final_dem_diff_snapshot.png*           | Snapshot plot of `final_dem_diff.tif`                                                    |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *initial_dem_diff_pdf.png*              | Plot of the probability density function of `initial_dem_diff`                           |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *final_dem_diff_pdf.png*                | Plot of the probability density function of `final_dem_diff`                             |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *initial_dem_diff_pdf.csv*              | Data of the probability density function of `initial_dem_diff`                           |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *final_dem_diff_pdf.csv*                | Data of the probability density function of `final_dem_diff`                             |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *initial_dem_diff_cdf.png*              | Plot of the cumulative density function of `initial_dem_diff`                            |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *final_dem_diff_cdf.png*                | Plot of the cumulative density function of `final_dem_diff`                              |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *initial_dem_diff_cdf.csv*              | Data of the cumulative density function of `initial_dem_diff`                            |
+-----------------------------------------+------------------------------------------------------------------------------------------+
| *final_dem_diff_cdf.csv*                | Data of the cumulative density function of `final_dem_diff`                              |
+-----------------------------------------+------------------------------------------------------------------------------------------+

The coregistration images saved with the ``coregistration`` ``save_optional_outputs`` option set to ``"True"``:

.. csv-table::
    :header: "Name","Description"
    :widths: auto
    :align: left

    *reproj_coreg_SEC.tif*,Reprojected and coregistered secondary DEM.
    *reproj_coreg_REF.tif*,Intermediate coregistered reference DEM.
    *reproj_SEC.tif*,Intermediate reprojected secondary DEM.
    *reproj_REF.tif*, Intermediate reprojected reference DEM.
    *nuth_kaab_iter#.png*,Iteration fit plot
    *ElevationDiff_AfterCoreg.png*,Elevation difference plot after coregistration
    *ElevationDiff_BeforeCoreg.png*,Elevation difference plot before coregistration

.. note::
    Both reprojected DEMs will have the secondary’s georeference grid.

.. note::
    Both coregistered DEMs will have the secondary’s georeference grid and an intermediate georeference origin.


Manual application of the coregistration offsets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If desired, the obtained **x** and **y** offsets may be manually applied using the following GDAL command with the obtained GDAL offset bounds:

.. code-block:: bash

    gdal_translate -a_ullr <ulx> <uly> <lrx> <lry> /PATH_TO/secondary_dem.tif /PATH_TO/coreg_secondary_dem.tif

Being *<ulx> <uly> <lrx> <lry>* the coordinate bounds of the offsets applied on **sec**. They are shown on logging the information after coregistration or stored in the **demcompare_results.json** file as **gdal_translate_bounds**.

Output directories
~~~~~~~~~~~~~~~~~~

With the command line execution, the following directories that may store the respective files will be automatically generated. The data that the directories can contain is also indicated.


.. code-block:: bash

    .output_dir
    +-- demcompare_results.json
    +-- sample_config.json
    +-- initial_dem_diff.tif
    +-- initial_dem_diff_snapshot.png
    +-- final_dem_diff.tif
    +-- final_dem_diff_snapshot.tif
    +-- stats
    |   +-- final_dem_diff_cdf.csv
    |   +-- final_dem_diff_cdf.png
    |   +-- initial_dem_diff_cdf.csv
    |   +-- initial_dem_diff_cdf.png
    |   +-- final_dem_diff_pdf.csv
    |   +-- final_dem_diff_pdf.png
    |   +-- initial_dem_diff_pdf.csv
    |   +-- initial_dem_diff_pdf.png
    |   <classification_layer_name*>
            +-- stats for each mode
    +-- coregistration
        +-- coreg_SEC.tif
        +-- reproj_REF.tif
        +-- reproj_DEM.tif
        +-- reproj_coreg_SEC.tif
        +-- reproj_coreg_REF.tif
        +-- nuth_kaab_tmp_dir
            +-- nuth_kaab_iter#*.png
            +-- ElevationDiff_AfterCoreg.png
            +-- ElevationDiff_BeforeCoreg.png

.. note::
    Please notice that some data will be missing or some directories will be empty if the required parameters are not activated.


References
**********

For the Nuth & Kääb universal coregistration algorithm :

.. [NuthKaab] Nuth, C. Kääb, 2011. A. Co-registration and bias corrections of satellite elevation data sets for quantifying glacier thickness change. Cryosphere 5, 271290.
