#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Chloe Thenoz (Magellium), Lisa Vo Thanh (Magellium).
#
# This file is part of mesh3d
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
Rational Polynomial Coefficients (RPC) tools
"""

import xml.etree.ElementTree as ET
from typing import Union

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000


class RPC:
    """Rational Polynomial Coefficients"""

    def __init__(
        self,
        polynomials: Union[list, tuple, np.ndarray],
        degrees: Union[list, tuple, np.ndarray],
        rpc_type: str,
    ):

        if len(polynomials) != 4:
            raise ValueError("Output dimensions should be 4 (P1, Q1, P2, Q2)")

        for i in range(len(polynomials)):
            if len(degrees) != len(polynomials[i]):
                raise ValueError(
                    "Number of monomes in the degrees array shall equal number of coefficients in the polynomial array."
                )

        if rpc_type not in ["DIRECT", "INVERSE"]:
            raise ValueError(
                f"RPC type should either be 'DIRECT' or 'INVERSE'."
            )

        self.polynomials = polynomials
        self.degrees = degrees
        self.rpc_type = rpc_type
        self.ref_offset = []
        self.ref_scale = []
        self.out_offset = []
        self.out_scale = []

    def set_normalisation_coefs(
        self, coefs: Union[list, tuple, np.ndarray]
    ) -> None:
        """Set normalisation coefficients for RPC"""
        if len(coefs) != 10:
            raise ValueError(
                "Normalisation and denormalisation coefficients shall be of size 10"
            )

        if self.rpc_type == "INVERSE":
            self.ref_offset = [coefs[1], coefs[3], coefs[5]]
            self.ref_scale = [coefs[0], coefs[2], coefs[4]]
            self.out_offset = [coefs[7], coefs[9]]
            self.out_scale = [coefs[6], coefs[8]]
        else:
            raise NotImplementedError


class PleiadesRPC(RPC):
    """RPC for Pleiades"""

    def __init__(
        self,
        rpc_type: str,
        polynomials: Union[list, tuple, np.ndarray] = None,
        path_rpc: str = None,
    ):

        self.degrees = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2],
            [1, 1, 1],
            [3, 0, 0],
            [1, 2, 0],
            [1, 0, 2],
            [2, 1, 0],
            [0, 3, 0],
            [0, 1, 2],
            [2, 0, 1],
            [0, 2, 1],
            [0, 0, 3],
        ]

        if path_rpc is not None:
            if rpc_type == "INVERSE":
                self.set_inverse_rpc(path_rpc)
            else:
                raise NotImplementedError

        elif polynomials is None:
            raise ValueError(
                "Either a valid RPC path or polynomials should be given in input."
            )

        else:
            super().__init__(polynomials, self.degrees, rpc_type)

    def _parse_rpc_xml(self, path_inverse_rpc: str) -> tuple:
        """
        Function that parses the xml file to get the RPC
        """
        tree = ET.parse(path_inverse_rpc)
        root = tree.getroot()

        coefs = []
        for coef_inv in root.iter("Inverse_Model"):
            for coef in coef_inv:
                coefs.append(float(coef.text))

        # coefs normalisation and denormalisation:
        # [long_scale, long_offset, lat_scale, lat_offset, alt_scale, alt_offset, samp_scale,
        # samp_offset, line_scale, line_offset]
        for coef_other in root.iter("RFM_Validity"):
            for coef in coef_other:
                if coef.tag == "LONG_SCALE":
                    long_scale = float(coef.text)
                if coef.tag == "LONG_OFF":
                    long_offset = float(coef.text)
                if coef.tag == "LAT_SCALE":
                    lat_scale = float(coef.text)
                if coef.tag == "LAT_OFF":
                    lat_offset = float(coef.text)
                if coef.tag == "HEIGHT_SCALE":
                    alt_scale = float(coef.text)
                if coef.tag == "HEIGHT_OFF":
                    alt_offset = float(coef.text)
                if coef.tag == "SAMP_SCALE":
                    samp_scale = float(coef.text)
                if coef.tag == "SAMP_OFF":
                    samp_offset = float(coef.text)
                if coef.tag == "LINE_SCALE":
                    line_scale = float(coef.text)
                if coef.tag == "LINE_OFF":
                    line_offset = float(coef.text)

        # Change image convention from (1, 1) to (0.5, 0.5)
        return (
            coefs[0:20],
            coefs[20:40],
            coefs[40:60],
            coefs[60:80],
            [
                long_scale,
                long_offset,
                lat_scale,
                lat_offset,
                alt_scale,
                alt_offset,
                samp_scale,
                samp_offset - 0.5,
                line_scale,
                line_offset - 0.5,
            ],
        )

    def set_inverse_rpc(self, path_rpc: str) -> None:
        """Set RPC for inverse location from a XML RPC file"""
        P1, Q1, P2, Q2, coefs = self._parse_rpc_xml(path_rpc)

        self.polynomials = [P1, Q1, P2, Q2]
        self.rpc_type = "INVERSE"
        self.set_normalisation_coefs(coefs)


def apply_rpc_list(
    rpc: RPC, input_coords: Union[tuple, list, np.ndarray]
) -> np.ndarray:
    """
    Function that computes inverse locations using rpc

    Parameters
    ----------
    rpc: RPC
        RPC parameters
    input_coords: (N, 3) or (N, 2) tuple or list or np.ndarray
        Coordinates expressed in geo (lon, lat) if ground coordinates, or (row, col) for image coordinates

    Returns
    -------
    res: (N, 3) or (N, 2) tuple or list or np.ndarray
        Coordinates transformed by direct (lon, lat, alt) or inverse (col, row) location
    """
    # normalize input
    norm_input = (np.array(input_coords) - np.array(rpc.ref_offset)) / np.array(
        rpc.ref_scale
    )

    result = []
    for i in range(len(rpc.polynomials)):
        val = np.zeros(len(input_coords))
        for j in range(len(rpc.polynomials[0])):
            monomial = np.ones(len(input_coords))
            for k in range(3):
                monomial *= np.power(norm_input[:, k], rpc.degrees[j][k])
            val += rpc.polynomials[i][j] * monomial
        result.append(val)

    if 0 in result[1][:] or 0 in result[3][:]:
        raise ValueError("Dividing by zero is not allowed")

    # [XNorm = P1/Q1, YNorm = P2/Q2]
    output = [result[0][:] / result[1][:], result[2][:] / result[3][:]]
    res = [[output[0][i], output[1][i]] for i in range(len(output[0]))]

    # denormalize output
    res = (np.array(res) * np.array(rpc.out_scale)) + np.array(rpc.out_offset)

    return res
