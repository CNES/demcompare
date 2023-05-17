<div align="center">
  <a href="https://github.com/CNES/demcompare"><img src="docs/source/images/demcompare_picto.png" alt="Demcompare" title="Demcompare"  width="200" align="center"></a>

<h4 align="center">Demcompare, a DEM comparison tool  </h4>

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](https://demcompare.readthedocs.io/en/latest/developer_guide/contributing.html)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0/)
[![Docs](https://readthedocs.org/projects/demcompare/badge/?version=latest)](https://demcompare.readthedocs.io/)

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#install">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#documentation">Documentation</a> •
</p>
</div>

## Overview

Demcompare is a python software that aims at **comparing two DEMs** together.

A DEM is a 3D computer graphics representation of elevation data to represent terrain.

**Demcompare** has several characteristics:

* Works whether or not the two DEMs share common format projection system, planimetric resolution, and altimetric unit.
* Performs the coregistration based on the Nuth & Kääb universal coregistration method.
* Offers two coregistration modes to choose which of both DEMs is to be adapted during coregistration.
* Provides a wide variety of standard metrics which can be classified.
* Classifies the stats by slope ranges by default, but one can provide any other data to classify the stats.

## Install

Only **Linux Plaforms** are supported (virtualenv or bare machine) with **Python >= 3.8** installed.

Demcompare is available on Pypi and can be typically installed through a [virtualenv](https://docs.python.org/3/library/venv):

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install demcompare
```

## Usage

Download the data samples and run the python script **demcompare** with sample configuration:

```bash
# download data samples
wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/srtm_blurred_and_shifted.tif
wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/srtm_ref.tif

# download demcompare predefined configuration
wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/sample_config.json

# run demcompare
demcompare sample_config.json
```

A report can be observed with:

```
firefox test_output/report/published_report/html/index.html
```

## Documentation

Please consult [our online documentation](https://demcompare.readthedocs.io).

## Licensing

Demcompare software is distributed under the Apache Software License (ASL) v2.0.

See [LICENSE](./LICENSE) file or <http://www.apache.org/licenses/LICENSE-2.0> for details.

Copyrights and authoring can be found in [NOTICE](./NOTICE) file.

## Related tools

[CARS](https://github.com/CNES/CARS) - CNES 3D reconstruction software

[Pandora](https://github.com/CNES/pandora) - CNES Stereo Matching framework
