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

"""
Class associated to Mesh 3D state machine
"""

"""
Notes à retirer:
- may_<trigger_name> pour vérifier que la transition est possible
"""

from typing import Callable

from loguru import logger
from transitions import Machine, MachineError


class Mesh3DObject(object):
    pass


class Mesh3DMachine(Machine):

    def __init__(self, dict_pcd_mesh: dict, initial_state: str = "initial_pcd") -> None:
        # Init arguments
        self.dict_pcd_mesh = dict_pcd_mesh
        self.initial_state = initial_state

        # Available states
        self.states_ = ["initial_pcd", "filtered_pcd", "denoised_pcd", "meshed_pcd", "textured_pcd"]

        # Available transitions
        self.transitions_ = [
            # From the 'initial_pcd' state
            {
                "trigger": "filter",
                "source": "initial_pcd",
                "dest": "filtered_pcd",
                "after": "filter_run"
            },
            {
                "trigger": "denoise_pcd",
                "source": "initial_pcd",
                "dest": "denoised_pcd",
                "after": "denoise_pcd_run"
            },
            {
                "trigger": "mesh",
                "source": "initial_pcd",
                "dest": "meshed_pcd",
                "after": "mesh_run"
            },

            # From the 'filtered_pcd' state
            {
                "trigger": "filter",
                "source": "filtered_pcd",
                "dest": "filtered_pcd",
                "after": "filter_run"
            },
            {
                "trigger": "denoise_pcd",
                "source": "filtered_pcd",
                "dest": "denoised_pcd",
                "after": "denoise_pcd_run"
            },
            {
                "trigger": "mesh",
                "source": "filtered_pcd",
                "dest": "meshed_pcd",
                "after": "mesh_run"
            },

            # From the 'denoised_pcd' state
            {
                "trigger": "denoise_pcd",
                "source": "denoised_pcd",
                "dest": "denoised_pcd",
                "after": "denoise_pcd_run"
            },
            {
                "trigger": "mesh",
                "source": "denoised_pcd",
                "dest": "meshed_pcd",
                "after": "mesh_run"
            },

            # From the 'meshed_pcd' state
            {
                "trigger": "denoise_mesh",
                "source": "meshed_pcd",
                "dest": "meshed_pcd",
                "after": "denoise_mesh_run"
            },
            {
                "trigger": "texture",
                "source": "meshed_pcd",
                "dest": "textured_pcd",
                "after": "texture_run"
            }
            # # From the "textured_pcd" state
            # {
            #     "trigger": "reset",
            #     "source": "textured_pcd",
            #     "dest": f"{self.initial_state}",
            # }

        ]

        # Initialize a machine model
        Machine.__init__(
            self,
            states=self.states_,
            initial=self.initial_state,
            transitions=self.transitions_,
            auto_transitions=False,
        )

    def run(self, input_step: str, cfg: dict) -> None:
        """
        Run mesh 3d step by triggering the corresponding machine transition

        Parameter
        --------
        input_step: str
            Name of the step to trigger
        cfg: dict
            Configuration dictionary
        """

        try:
            self.trigger(input_step, cfg)

        except (MachineError, KeyError, AttributeError):
            logger.error(f"A problem occurs during Pandora running {input_step} step. Be sure of your sequencing.")
            raise

    def filter_run(self, cfg: dict, check_mode: bool = False):
        """Filter the point cloud from outliers"""

        if check_mode:
            # For checking transition validity
            return

        else:
            from core import filter




    def denoise_pcd_run(self, cfg: dict, check_mode: bool = False):
        if check_mode:
            return
        else:
            pass

    def mesh_run(self, cfg: dict, check_mode: bool = False):
        if check_mode:
            return
        else:
            pass

    def denoise_mesh_run(self, cfg: dict, check_mode: bool = False):
        pass

    def texture_run(self, cfg: dict, check_mode: bool = False):
        if check_mode:
            return
        else:
            logger.debug("Bonjour moi je textuuuuuuuure lourd lourd")

    def check_transitions(self, cfg: dict):
        """
        Browse user defined steps and pass them just to check they are valid steps in the state machine.
        No action is done.
        """

        logger.debug("Check state machine transitions' validity.")

        try:
            for k, step in enumerate(cfg["state_machine"]):
                self.trigger(step['action'], cfg, check_mode=True)

        except (MachineError, KeyError, AttributeError):
            logger.error(f"A problem occurs during Mesh 3D transition check. Be sure of your sequencing.")
            raise

        # Add transition to reset
        self.add_transition(trigger="reset", source=self.state, dest=f"{self.initial_state}")

        # Reset at initial step
        self.trigger("reset", cfg)

        # Remove transition for resetting
        self.remove_transition(trigger="reset", source=self.state, dest=f"{self.initial_state}")
