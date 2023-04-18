Contributing guide
==================

Bug report
**********

Any proven or suspected malfunction should be traced in a bug report, the latter being an issue in the **demcompare** repository.

.. note::
  Please open a bug report: Notifying the potential bugs is the first way to contribute to the software !

In the problem description, be as accurate as possible. Include:
 - The procedure used to initialize the environment
 - The incriminated command line or python function
 - The content of the input and output configuration files (*content.json*)

Contributing workflow
*********************

Any code modification requires a Merge Request. It is forbidden to push patches directly into master (this branch is protected).

It is recommended to open your Merge Request as soon as possible in order to inform the developers of your ongoing work.
Please add *WIP:* before your Merge Request title if your work is in progress: This prevents an accidental merge and informs the other developers of the unfinished state of your work.

The Merge Request shall have a short description of the proposed changes. If it is relative to an issue, you can signal it by adding *Closes xx* where xx is the reference number of the issue.

Likewise, prefix the associated branch's name by *xx-* in order to link it to the xx issue (ie. 999-name_branch_example).

**Demcompare**'s Classical workflow is :
 - Check Licence and sign :ref:`contribution_license_agreement` (Individual or Corporate)
 - Create an issue (or begin from an existing one)
 - Create a Merge Request from the issue: a MR is created accordingly with *WIP:*, *Closes xx* and associated *xx-name-issue* branch
 - Modify **demcompare**'s code from a local working directory or from the forge (less possibilities)
 - Git add, commit and push from local working clone directory or from the forge directly
 - Follow `Conventional commits <https://www.conventionalcommits.org/>`_ specifications for commit messages
 - Beware that pre-commit hooks can be installed for code analysis (see below pre-commit validation).
 - Run the tests with pytest on your modifications (or don't forget to add ones).
 - When finished, change your Merge Request name (erase *WIP:* in title ) and ask to review the code.


.. _contribution_license_agreement:

Licensing
*********

**Demcompare** requires that contributors sign out a `Contributor License Agreement <https://en.wikipedia.org/wiki/Contributor_License_Agreement>`_.

To accept your contribution, we need you to complete, sign and email to *cars@cnes.fr* an
`Individual Contributor Licensing Agreement <https://raw.githubusercontent.com/CNES/demcompare/master/docs/source/CLA/ICLA-DEMCOMPARE.doc>`_ (ICLA) form 
or a `Corporate Contributor Licensing Agreement <https://raw.githubusercontent.com/CNES/demcompare/master/docs/source/CLA/CCLA-DEMCOMPARE.doc>`_ (CCLA) form
if you are contributing on behalf of your company or another entity which retains copyright for your contribution.

The copyright owner (or owner's agent) must be mentioned in headers of all
modified source files and also added to the `AUTHORS file <https://github.com/CNES/demcompare/blob/master/AUTHORS.md>`_.

Merge request acceptation process
*********************************

Two Merge Requests types to help the workflow : 
- Simple merge requests (bugs, documentation) can be merged directly by the author with rights on master. 
- Advanced merge requests (typically a big change in code) are flagged with “To be Reviewed” by the author

This mechanism is to help quick modifications and avoid long reviews on unneeded simple merge requests. The author has to be responsible in the
need or not to be reviewed.

The Advanced Merge Request will be merged into master after being reviewed by a demcompare steering committee (core committers) composed of:

* Emmanuelle Sarrazin (CNES) 
* Emmanuel Dubois (CNES)

Only the members of this committee can merge into master for advanced merge requests.

The checklist of an Advanced Merge Request acceptance is the following:

*  At least one code review have been done by members of the group above (who check among others the correct application of the rules listed in the :doc:`coding_guide`). 
*  All comments of the reviewers has been dealt with and are closed 
*  The reviewers have signaled their approbation (thumb up) 
*  No developer is against the Merge Request (thumb down)
