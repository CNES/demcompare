# Changelog

Only the first "Unreleased" section of this file corresponding of next release can be updated along the development of each new, changed and fixed features.
When publication of a new release, the section "Unreleased" is blocked to the next chosen version and name of the milestone at a given date.
A new section Unreleased is opened then for next dev phase.

## Unreleased

### Added

### Changed

### Fixed

## 0.4.0 Bugs

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
