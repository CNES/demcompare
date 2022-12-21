<div align="center">
  <a href="https://github.com/CNES/demcompare"><img src="docs/source/images/demcompare_picto.png" alt="Demcompare" title="Demcompare"  width="300" align="center"></a>

<h4 align="center">DEMcompare, a DEM comparison tool  </h4>

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0/)
[![Docs](https://readthedocs.org/projects/demcompare/badge/?version=latest)]('https://demcompare.readthedocs.io/?badge=latest)

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#install">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#outputs-processing">Outputs processing</a> •
  <a href="#references">References</a>
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

Download our data sample and run the python script **demcompare**:


```bash
# download data samples
wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/images/srtm_sample.zip  # input stereo pair
wget https://raw.githubusercontent.com/CNES/demcompare/master/data_samples/json_conf_files/nuth_kaab_config.json # configuration file

# uncompress data
unzip srtm_sample.zip

#run demcompare
demcompare nuth_kaab_config.json
```

The results can be observed with:

```
    firefox test_output/doc/published_report/html/demcompare_report.html &
```

## To go further

Please consult [our online documentation](https://demcompare.readthedocs.io).

You will learn:
- Which steps you can [use and combine](https://demcompare.readthedocs.io/en/latest/userguide/step_by_step.html)
- How to use the [command line execution](https://demcompare.readthedocs.io/en/latest/userguide/command_line_execution.html)
- Which parameters you can set in the [input configuration](https://demcompare.readthedocs.io/en/latest/userguide/inputs.html)


## Licensing

demcompare software is distributed under the Apache Software License (ASL) v2.0.

See [LICENSE](./LICENSE) file or http://www.apache.org/licenses/LICENSE-2.0 for details.

Copyrights and authoring can be found in [NOTICE](./NOTICE) file.

## Related

[CARS](https://github.com/CNES/CARS) - CNES 3D reconstruction software

[Pandora](https://github.com/CNES/pandora) - CNES Stereo Matching framework
