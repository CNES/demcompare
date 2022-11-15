#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2022 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of demcompare
# (see https://github.com/CNES/demcompare).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Init file for classification_layer module.
Imports are used to simplify calls to module API ClassificationLayer.
"""
# Demcompare imports
from . import segmentation_classification, slope_classification
from .classification_layer import ClassificationLayer
from .fusion_classification import FusionClassificationLayer
from .global_classification import GlobalClassificationLayer

__all__ = [
    "segmentation_classification",
    "slope_classification",
    "ClassificationLayer",
    "FusionClassificationLayer",
    "GlobalClassificationLayer",
]
# To avoid flake8 F401
