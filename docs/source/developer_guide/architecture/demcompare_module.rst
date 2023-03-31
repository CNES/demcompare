.. _demcompare_module:


Demcompare module
=================

This describes the main demcompare module in `__init__.py` file which orchestrates the demcompare API from an input configuration file.

Demcompare module's orchestration for coregistration step
---------------------------------------------------------

.. figure:: /images/coreg_api.png
    :width: 800px
    :align: center

    Demcompare's orchestration for coregistration step.


To perform the dems coregistration, demcompare's module performs the following steps:

1. Loads the input dems using the **dem_tools** module's **load_dem** function.
2. Creates a **Coregistration** object and obtains the dem's transformation object using the **coregistration**'s **compute_coregistration** function.
3. Applies the obtained transformation to the secondary dem using the **transformation**'s **apply_transform** function.

Demcompare module's orchestration for statistics step
-----------------------------------------------------

.. figure:: /images/stats_api.png
    :width: 800px
    :align: center

    Demcompare's orchestration for statistics step.

To perform the dems statistics, demcompare's module performs the following steps:

1. Loads the input dems using the **dem_tools** module's **load_dem** function.
2. Reprojects both dems to the same size and resolution using **dem_tools** module's **reproject_dems** function.

.. note::

    If coregistration has previously been done, the **coregistration**'s objects internal dems called **reproj_coreg_ref** and **reproj_coreg_sec** are used for the altitude difference computation, so that no manual reprojection needs to be done. Please see :ref:`statistics` "With coregistration step" section for more details.

3. Computes the altitude difference dem using the **dem_tools** module's **compute_alti_diff_for_stats** function.
4. Creates a **Stats_processing** object and obtains the **stats_dataset** using the **stats_processing**'s **compute_stats** function.


Module files description
************************

- **__init__.py**

This module includes demcompare's run function, which performs the input cfg's steps.

- **helpers_init.py**

In this module high level parameters are checked and default options are set. Some helper functions to handle
the output paths from the __init__ are also included here.

- **log_conf.py**

The logconf module in demcompare contains logging configuration functions.

- **output_tree_design.py**

Module containing the default output tree design architecture for demcompare's output directory.

Add configuration handling.