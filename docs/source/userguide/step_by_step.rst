.. _step_by_step:

<<<<<<< HEAD

Step by step
============

The following sections describe demcompare's DSM comparison steps.
=======
Step by step
============

The following sections describe Demcompare's DSM comparaison steps.
>>>>>>> parent of d08274f... doc: uniform demcompare naming, modify getting_started

.. toctree::

    step_by_step/coregistration.rst
    step_by_step/statistics.rst
    step_by_step/report.rst

<<<<<<< HEAD

1. During the optional coregistration step, demcompare performs the Nuth et Kaab coregistration on two uncoregistered DEMs like the ones below :

.. image:: /images/doc_ref.gif
    :width: 300px

In this example, the two uncoregistered DEMs had the initial altitude difference shown on the left image. After Nuth et Kaab coregistration, demcompare obtains the final altitude difference shown on the right image:

.. image:: /images/initial_dh.png
    :width: 260px

.. image:: /images/final_dh.png
    :width: 260px

2. Once the DSMs are coregistered, demcompare is ready to compare both DEMs computing a wide variety of standard metrics and statistics.

3. A report to better visualize the obtained statistics may be generated.
=======
Demcompare allows one to execute only a subset of the whole process. As such, a `--step` command line argument is
provided. It accepts values in `coregistration` `stats` `report` :

.. code-block:: bash

    [user@machine] $ demcompare
    usage: demcompare [-h]
      [--step step_name [step_name ...]]
      [--display] [--version]
      config.json

- All the steps but **stats** are optional

- Demcompare can start at any step as long as previously required steps have been launched.

  - This means that one can launch the report step only as long as the stats step has already been performed from a previous Demcompare launch and the *config.json* remains the same. The steps are space-separated (no comma).

.. note::  Coregistration step is not mandatory for stats and following steps as one can decide its DEMs are already coregistered.

.. note::  Be careful: the positional config.json can be wrongly used as a step if used after the `--step` option.

.. code-block:: bash

    demcompare --step stats config.json : KO
    demcompare config.json --step stats : OK
>>>>>>> parent of d08274f... doc: uniform demcompare naming, modify getting_started
