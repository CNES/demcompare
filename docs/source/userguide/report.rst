.. _report:

Generated output report
=======================

If the **demcompare** execution includes the **statistics** step and the output directory has been specified, then the output `<test_output>/doc/published_report/` directory contains a full generated sphinx documentation with the results presented
for each mode and each class, in html or latex format.

Once **demcompare** has been executed, the report can be observed using:

.. code-block:: bash

    firefox test_output/doc/published_report/html/demcompare_report.html &

Report's modular structure
--------------------------

The output report has the following structure. Notice that building a report with only the stats results of both parts is possible :

1. Coregistration results
    1.1 Without coregistration:
        - Initial elevation difference image *initial_dem_diff.tif*
        - Initial elevation difference cumulative probability
        - Initial elevation difference histogram
    1.2 With coregistration:
        - Final elevation difference image *final_dem_diff.tif*
        - Final elevation difference cumulative probability
        - Final elevation difference histogram

2. Stats results
    2.1 For each classification layer. Notice that a classification layer may not have intersection and exclusion modes :
        2.1.1 Mode: standard
            - Table showing comparison metrics
        2.1.2 Mode: intersection-classification
            - Table showing comparison metrics
        2.1.3 Mode: exclusion-classification
            - Table showing comparison metrics

