#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2022 Chloe Thenoz (Magellium), Lisa Vo Thanh (Magellium).
#
# This file is part of mesh_3d
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

"""Global utils"""

import subprocess


def timer_display(elapsed_time):
    """
    Format time in hh:mm:ss
    """
    (t_min, t_sec) = divmod(round(elapsed_time, 3), 60)
    (t_hour, t_min) = divmod(t_min, 60)
    return "{}:{}:{}".format(
        str(round(t_hour)).zfill(2),
        str(round(t_min)).zfill(2),
        str(round(t_sec)).zfill(2),
    )


def format_timer_display(elapsed_time):
    """
    Display timer as "HH:MM:SS" if the elapsed time is of at least 1s, otherwise print the real time in seconds
    """
    return (
        timer_display(elapsed_time)
        if (elapsed_time) >= 1.0
        else "{:.6f} s".format(elapsed_time)
    )
