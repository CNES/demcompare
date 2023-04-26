.. _report:

Generated output report
=======================

.. warning::
  Demcompare report is work in progress ! 


If the **demcompare** execution includes the **statistics** step 
and the output directory has been specified, a report can be generated using the "report" step.

For now, only a HTML and PDF report can be generated from a specific sphinx source report. 

Two values are possible in configuration: 
- "default": equal to "sphinx" as the only report generator for now
- "sphinx": use demcompare sphinx report generator



.. code-block:: json

      "report": "default"


Sphinx report
*************

The output `<test_output>/report/published_report/` directory contains 
a full generated sphinx documentation with the results presented
for each mode and each class, in html or latex format.

The source of the sphinx report is in  `<test_output>/report/src``

Once **demcompare** has been executed with report configuration,
the report can be observed using a browser:

.. code-block:: bash

    firefox test_output/report/published_report/html/demcompare_report.html &

Report's modular structure
--------------------------

The output report has the following structure.
Notice that building a report with only the stats results of both parts is possible :

1. Coregistration results
    1.1 Initial elevations: 
        - Initial elevation difference image *initial_dem_diff.tif*
        - Initial elevation difference histogram
        - Initial elevation difference cumulative probability
    1.2 Final elevations after coregistration:
        - Final elevation difference image *final_dem_diff.tif*
        - Final elevation difference histogram
        - Final elevation difference cumulative probability

2. Stats results
    2.1 For each classification layer. Notice that a classification layer may not have intersection and exclusion modes :
        2.1.1 Mode: standard
            - Table showing comparison metrics
        2.1.2 Mode: intersection-classification
            - Table showing comparison metrics
        2.1.3 Mode: exclusion-classification
            - Table showing comparison metrics

