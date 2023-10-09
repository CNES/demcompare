#!/usr/bin/env python
# coding: utf8
#
# Copyright (C) 2023 CNES.
#
# This file is part of cars-mesh

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
Class associated to CARS-MESH state machine
"""

# Standard imports
import logging

# Third party imports
from transitions import Machine, MachineError

# Cars-mesh imports
from . import param
from .tools.handlers import Mesh


class CarsMeshMachine(Machine):
    """CARS-MESH state machine"""

    def __init__(
        self, mesh_data: Mesh, initial_state: str = "initial_pcd"
    ) -> None:
        """
        Init a state machine for 3D mesh reconstruction

        Parameters
        ----------
        mesh_data: Mesh
            Mesh instance to process
        initial_state: str (default='initial_pcd')
            Specify the initial state. Can either be 'initial_pcd' (if
            input data is a point cloud) or 'meshed_pcd' (if input data is a
            mesh)
        """
        # Init arguments
        self.mesh_data = mesh_data
        self.initial_state = initial_state

        # Available states
        self.states_ = [
            "initial_pcd",
            "filtered_pcd",
            "denoised_pcd",
            "meshed_pcd",
            "textured_pcd",
        ]

        # Available transitions
        self.transitions_ = [
            # From the 'initial_pcd' state
            {
                "trigger": "filter",
                "source": "initial_pcd",
                "dest": "filtered_pcd",
                "after": "filter_run",
            },
            {
                "trigger": "denoise_pcd",
                "source": "initial_pcd",
                "dest": "denoised_pcd",
                "after": "denoise_pcd_run",
            },
            {
                "trigger": "mesh",
                "source": "initial_pcd",
                "dest": "meshed_pcd",
                "after": "mesh_run",
            },
            # From the 'filtered_pcd' state
            {
                "trigger": "filter",
                "source": "filtered_pcd",
                "dest": "filtered_pcd",
                "after": "filter_run",
            },
            {
                "trigger": "denoise_pcd",
                "source": "filtered_pcd",
                "dest": "denoised_pcd",
                "after": "denoise_pcd_run",
            },
            {
                "trigger": "mesh",
                "source": "filtered_pcd",
                "dest": "meshed_pcd",
                "after": "mesh_run",
            },
            # From the 'denoised_pcd' state
            {
                "trigger": "denoise_pcd",
                "source": "denoised_pcd",
                "dest": "denoised_pcd",
                "after": "denoise_pcd_run",
            },
            {
                "trigger": "mesh",
                "source": "denoised_pcd",
                "dest": "meshed_pcd",
                "after": "mesh_run",
            },
            # From the 'meshed_pcd' state
            {
                "trigger": "simplify_mesh",
                "source": "meshed_pcd",
                "dest": "meshed_pcd",
                "after": "simplify_mesh_run",
            },
            {
                "trigger": "denoise_mesh",
                "source": "meshed_pcd",
                "dest": "meshed_pcd",
                "after": "denoise_mesh_run",
            },
            {
                "trigger": "texture",
                "source": "meshed_pcd",
                "dest": "textured_pcd",
                "after": "texture_run",
            },
        ]

        # Initialize a machine model
        Machine.__init__(
            self,
            states=self.states_,
            initial=self.initial_state,
            transitions=self.transitions_,
            auto_transitions=False,
        )

    def run(self, step: dict, cfg: dict) -> None:
        """
        Run CARS-MESH step by triggering the corresponding machine transition

        Parameters
        ----------
        step: str
            Name of the step to trigger
        cfg: dict
            Configuration dictionary
        """

        try:
            self.trigger(step["action"], step, cfg)

        except (MachineError, KeyError, AttributeError):
            logging.error(
                f"A problem occurs during CARS-MESH running {step['action']} "
                f"step. Be sure of your sequencing."
            )
            raise

    def filter_run(
        self, step: dict, cfg: dict, check_mode: bool = False
    ) -> None:
        """
        Filter the point cloud from outliers

        Parameters
        ----------
        step: dict
            Parameters of the step to run
        cfg: dict
            Configuration dictionary
        check_mode: bool (default=False)
            Option to run the transition checker
        """
        assert isinstance(cfg, dict)

        if check_mode:
            # For checking transition validity
            pass

        else:
            # Apply the filtering method chosen by the user
            self.mesh_data.pcd = param.TRANSITIONS_METHODS[step["action"]][
                step["method"]
            ](self.mesh_data.pcd, **step["params"])

    def denoise_pcd_run(
        self, step: dict, cfg: dict, check_mode: bool = False
    ) -> None:
        """
        Denoise the point cloud

        Parameters
        ----------
        step: dict
            Parameters of the step to run
        cfg: dict
            Configuration dictionary
        check_mode: bool (default=False)
            Option to run the transition checker
        """
        assert isinstance(cfg, dict)

        if check_mode:
            # For checking transition validity
            pass

        else:
            # Apply the denoising method chosen by the user
            self.mesh_data.pcd = param.TRANSITIONS_METHODS[step["action"]][
                step["method"]
            ](self.mesh_data.pcd, **step["params"])

    def mesh_run(self, step: dict, cfg: dict, check_mode: bool = False) -> None:
        """
        Mesh the point cloud

        Parameters
        ----------
        step: dict
            Parameters of the step to run
        cfg: dict
            Configuration dictionary
        check_mode: bool (default=False)
            Option to run the transition checker
        """
        assert isinstance(cfg, dict)

        if check_mode:
            # For checking transition validity
            pass

        else:
            # Apply the meshing method chosen by the user
            self.mesh_data = param.TRANSITIONS_METHODS[step["action"]][
                step["method"]
            ](self.mesh_data.pcd, **step["params"])

    def simplify_mesh_run(
        self, step: dict, cfg: dict, check_mode: bool = False
    ) -> None:
        """
        Simplify the mesh to reduce the number of faces

        Parameters
        ----------
        step: dict
            Parameters of the step to run
        cfg: dict
            Configuration dictionary
        check_mode: bool (default=False)
            Option to run the transition checker
        """
        assert isinstance(cfg, dict)

        if check_mode:
            # For checking transition validity
            pass

        else:
            # Apply the meshing method chosen by the user
            self.mesh_data = param.TRANSITIONS_METHODS[step["action"]][
                step["method"]
            ](self.mesh_data, **step["params"])

    def denoise_mesh_run(
        self, step: dict, cfg: dict, check_mode: bool = False
    ) -> None:
        """
        Denoise the mesh

        Parameters
        ----------
        step: dict
            Parameters of the step to run
        cfg: dict
            Configuration dictionary
        check_mode: bool (default=False)
            Option to run the transition checker
        """
        assert isinstance(cfg, dict)

        if check_mode:
            # For checking transition validity
            pass

        else:
            # Apply the meshing method chosen by the user
            self.mesh_data = param.TRANSITIONS_METHODS[step["action"]][
                step["method"]
            ](self.mesh_data, **step["params"])

    def texture_run(
        self, step: dict, cfg: dict, check_mode: bool = False
    ) -> None:
        """
        Texture the mesh

        Parameters
        ----------
        step: dict
            Parameters of the step to run
        cfg: dict
            Configuration dictionary
        check_mode: bool (default=False)
            Option to run the transition checker
        """
        assert isinstance(cfg, dict)

        if check_mode:
            # For checking transition validity
            pass

        else:
            # Apply the texturing method chosen by the user
            self.mesh_data = param.TRANSITIONS_METHODS[step["action"]][
                step["method"]
            ](self.mesh_data, cfg, **step["params"])

    def check_transitions(self, cfg: dict) -> None:
        """
        Browse user defined steps and pass them just to check they are valid
        steps in the state machine.
        No action is done.

        Parameters
        ----------
        cfg: dict
            Configuration dictionary
        """

        logging.debug("Check state machine transitions' validity.")

        try:
            # Check if the sequencing asked by the user is valid
            for _, step in enumerate(cfg["state_machine"]):
                self.trigger(step["action"], step, cfg, check_mode=True)

        except (MachineError, KeyError, AttributeError):
            logging.error(
                "A problem occurs during CARS-MESH transition check. "
                "Be sure of your sequencing."
            )
            raise

        # Add transition to reset
        self.add_transition(
            trigger="reset", source=self.state, dest=f"{self.initial_state}"
        )

        # Reset at initial step
        self.trigger("reset", cfg)

        # Remove transition for resetting
        self.remove_transition(
            trigger="reset", source=self.state, dest=f"{self.initial_state}"
        )
