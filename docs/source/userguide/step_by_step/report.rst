.. _report:

Generated output report
=======================

The output `<test_output>/doc/published_report/` directory contains a full generated sphinx documentation with all the results presented
for each mode and each set, in html or latex format.

Once **demcompare** has been executed, the report can be observed using:

.. code-block:: bash

    firefox test_output/doc/published_report/html/demcompare_report.html &

Report structure
****************

The output report has the following structure:

1. Elevation differences
    1.1 Without coregistration:
        - Initial elevation difference image *init_dh.tif*
        - Initial elevation difference cumulative probability
        - Initial elevation difference histogram
    1.2 With coregistration:
        - Final elevation difference image *final_dh.tif*
        - Final elevation difference cumulative probability
        - Final elevation difference histogram

2. Stats results
    2.1 Classification layer (one for the default *slope* layer and one for each classification layer defined by the user)
        2.1.1 Mode: standard
            - Graph showing mean and standard deviation
            - Fitted graph showing mean and standard deviation
            - Table showing comparison metrics
        2.1.2 Mode: coherent-classification
            - Graph showing mean and standard deviation
            - Fitted graph showing mean and standard deviation
            - Table showing comparison metrics
        2.1.3 Mode: incoherent-classification
            - Graph showing mean and standard deviation
            - Fitted graph showing mean and standard deviation
            - Table showing comparison metrics


