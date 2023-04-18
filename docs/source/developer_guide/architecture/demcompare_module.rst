.. _demcompare_module:


Demcompare module
=================

The main demcompare module in `demcompare_module file <https://github.com/CNES/demcompare/blob/master/demcompare/__init__.py>`_ orchestrates the demcompare
API from an input configuration file, hence, from an execution using the :ref:`demcompare_cli`.

The configuration file specifies digital elevation models inputs to compare and also the pipeline to execute.
This pipeline depends on the optional :ref:`coregistration` and/or :ref:`statistics` steps configuration.

If coregistration and/or statistics steps are to be computed,
then each step orchestration will be handled by demcompare module as follows:

Coregistration step orchestration
---------------------------------

.. figure:: /images/coreg_api.png
    :width: 800px
    :align: center

    Demcompare coregistration step orchestration


To perform the dems coregistration :ref:`coregistration`, demcompare's module performs the following steps:

1. Loads the input dems using the **dem_tools** module's *load_dem* function.
2. Creates a **Coregistration** object and obtains the dem's transformation object using the **coregistration**'s *compute_coregistration* function.
3. Applies the obtained transformation to the secondary dem using the **transformation**'s *apply_transform* function.

For more details on the coregistration modules architecture, please see :ref:`coregistration_modules`.

Statistics step orchestration
-----------------------------

.. figure:: /images/stats_api.png
    :width: 800px
    :align: center

    Demcompare's orchestration for statistics step.

To perform the dems statistics :ref:`statistics`, demcompare's module performs the following steps:

1. Loads the input dems using the **dem_tools** module's *load_dem* function.
2. Reprojects both dems to the same size and resolution using **dem_tools** module's *reproject_dems* function.
3. Computes the altitude difference dem using the **dem_tools** module's *compute_alti_diff_for_stats* function.
4. Creates a **Stats_processing** object and obtains the **stats_dataset** using the **stats_processing**'s *compute_stats* function.


.. note::

    If coregistration has previously been done, the **coregistration**'s objects internal dems called **reproj_coreg_ref** and **reproj_coreg_sec** are used for the altitude difference computation, so that no manual reprojection needs to be done. Please see :ref:`statistics` "With coregistration step" section for more details.


For more details on the statistics modules architecture, please see :ref:`stats_modules`.

Module files description
************************

- **helpers_init** module in `helpers_init.py file <https://github.com/CNES/demcompare/blob/master/demcompare/helpers_init.py>`_

.. _helpers_init:

In this module high level parameters of the input configuration are checked and default options are set when
not already defined. Some helper functions to handle the output paths from the ` are also included here.

- **log_conf** module in `log_conf.pyfile <https://github.com/CNES/demcompare/blob/master/demcompare/log_conf.py>`_

.. _log_conf:

The logconf module in demcompare contains logging configuration functions.

- **output_tree_design** module in `output_tree_design.py file <https://github.com/CNES/demcompare/blob/master/demcompare/output_tree_design.py>`_

.. _output_tree_design:

Module containing the default output tree design architecture for demcompare's output directory.
This module contains the functions to create the output tree directory and defines where each output file is to be
saved during a demcompare execution. By default, it considers that a pipeline execution with :ref:`coregistration` and
:ref:`statistics` is run.
