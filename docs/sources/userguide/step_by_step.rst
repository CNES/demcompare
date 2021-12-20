.. _step_by_step:

Step by step
============

The following sections describe Demcompare's DSM coregistration steps.

.. toctree::

    step_by_step/coregistration.rst
    step_by_step/statistics.rst
    step_by_step/report.rst

demcompare allows one to execute only a subset of the whole process. As such, a --step command line argument is
provided. It accepts values in "coregistration" "stats" "report" :

.. code-block:: bash
    [user@machine] $ demcompare
    usage: demcompare [-h]
      [--step step_name [step_name ...]]
      [--display] [--version]
      config.json

All the steps but stats are optional

demcompare can start at any step as long as previously required steps have been launched.

This means that one can launch the report step only as long as the stats step has already been performed from a previous
demcompare launch and the config.json remains the same.

coregistration step is not mandatory for stats and following steps as one can decide its DEMs are already coregistered.
The steps are space-separated (no comma)
Be careful: the positional config.json can be wrongly used as a step if used after the --step option.

.. code-block:: bash
    demcompare --step stats config.json : KO
    demcompare config.json --step stats : OK