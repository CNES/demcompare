# Changelog

Only the first "Unreleased" section of this file corresponding of next release can be updated along the development of each new, changed and fixed features.
When publication of a new release, the section "Unreleased" is blocked to the next chosen version and name of the milestone at a given date.
A new section Unreleased is opened then for next dev phase.

## Unreleased

### Added

### Changed

### Fixed

## 0.5.2 Developer guide, fix bugs (June 2023)

### Added

- add developer guide and architecture description [#146]

### Changed

- adapt with coregistration and clean snapshots report outputs [#207,#209]

### Fixed

- bug colorbar with nan [#206] 
- bug stats_dataset duplicated mode [#193]
- bug report not shown anymore [#192]
- add mypy in make lint for CI [#191]
- typos [#188, #189]

## 0.5.1 quick bugs, add version on cli (March 2023)

### Added

- add version in cli argument [#185]

### Changed

- update and clean version in Makefile [#184, #187]

### Fixed

- Fix python3.10 warnings [#183]
- Bug readthedocs typo [#182]
- pylint and black errors with upgrade [#187]

## 0.5.0 Refactoring with new API (December 2022)

### Added

- Add demcompare general notebook [#109]
- Add coregistration notebook for demcompare users. [#107]
- Add statistics notebook for demcompare users.[#108]
- Refactoring validation plan. [#61]
- update test headers from validation plan [#140, #138, #139, #137, #132, #133, #134, #135, #136, #141, #142, #173, #175]
- Add validation plan tests [#123, #124]
- Add functional tests list [#143]
- Add static type checker mypy [#94]
- Demcompare user notebooks (coregistration, stats) [#107, #108, #109]
- add Authors file [#150]

### Changed

- Refactoring design and conception [#74, #69]
- Refactoring image and dem tools with demcompare dataset. [#76]
- Refactoring coregistration module. [#75]
- Refactoring statistics module. [#77]
- Refactoring init demcompare. [#78]
- Clean refactoring [#104]
- refactoring logs and use logging only [#63]
- Update user documentation. [#81, #130, #129, #125]
- refactoring output_dir usage [#127]
- first experimental report refactoring with dash [#52]
- upgrade to python 3.8 [#96]
- upgrade setuptools_scm [#99]
- upgrade python > 3.8 [#169]

### Fixed

- Correct pixel filtering and add exception when outside of the geoid scope. [#86]
- Fix/clean TODOs left in code [#97]
- Fix code quality isort, black, pylint and flake8 [#92, #179, #163]
- Fix bug in setuptools version in pip install -e mode [#119]
- Fix reported planimetry 2D shift in output [#101]
- Fix pytest warnings [#93]
- Fix bounds used in demcompare for notebooks [#156]
- Clean packaging, makefile, upgrade python>=3.8 [#168, #169]
- Bounds dans le dataset alti_diff [#156]
- Fix bugs for same dems or nan dems in inputs [#88]

## 0.4.0 Last old release (September 2022)

### Added

- add gitlab template release [#83]

### Changed

### Fixed

- linting errors [#95, #100]
- flake8-copyright linting bug : desactivation [#102]
- align rmse rounding with all stats [#103]

## 0.3.0 Clean bugs, tests and documentation Release (April 2022)

### Added

- Add pytest tests. [#23]
- Add sphinx doc and readthedocs. [#30]
- Clarifications and make optional input parameter georef. [#71]
- Give bounds to coregister DEM with GDAL and clarifications on coregDEM and coregREF. [#70]

### Changed

- Limit decimal number on output offsets. [#62]
- Suppress the 0.5 offset on translate function. [#36]
- Suppress unused mosaic tool.[#68]

### Fixed

- Correct classification layers coregistration. [#58]
- Correct input DSM ROI not being considered. [#41]
- Fix negative percentil on histogram creation. [#72]
- Filter Nuth et Kaab zero division. [#53]

## 0.2.0 Fix Bugs, Clean and small functionalities Release (September 2021)

### Added

- Add Contributor Licence Agreement. [#37]
- Add cumulative Density Function stats [#34]
- Add Nuth/Kaab iterations number option [#13]
- Add option to give a local geoid model. [#33]
- Add sonarqube configuration. [#46]
- Add logo in Readme [#32]

### Changed

### Fixed

- Fix numpy, cython pip upgrade install [#37]
- Fix remove_outliers no_data 0 bug. [#43]
- Clean Makefile [#44]
- Fix install numpy, cython and upgrade pip [#35]
- Fix input images orientation possible bug [#49]

## 0.1.0 First Open Source Official Release (July 2020)

### Added

- Dem comparison python3 tool publication
- Nuth et Kaab algorithm coregistration
- Generated reports with stats (pdf, html through sphinx)
- Documentation basics (README)
- Minimal tests data (tests directory)
- Apache 2 license
