.. _step_by_step:

.. role:: bash(code)
   :language: bash

Step by step
============

The following sections describe **demcompare**'s DSM comparaison steps.

.. toctree::

    step_by_step/coregistration.rst
    step_by_step/statistics.rst
    step_by_step/report.rst

**Demcompare**'s execution performs the following comparison steps:

1. During the optional coregistration step, **demcompare** performs the [NuthKaab]_ **coregistration** on two uncoregistered DEMs like the ones below :

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

After Nuth et Kaab coregistration, **demcompare** obtains the final altitude difference shown on the following image:

.. figure:: /images/final_dh.png
    :width: 260px
    :align: center

    Final altitude difference between the two coregistered DSMs.

2. Once the DSMs are coregistered, **demcompare** is ready to compare both DEMs computing a wide variety of standard metrics and **statistics**.

3. A **report** to better visualize the obtained statistics may be generated.
