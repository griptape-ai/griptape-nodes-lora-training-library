from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from griptape_nodes.exe_types.core_types import Parameter

from train_lora.model_family_parameters import TrainLoraModelFamilyParameters

if TYPE_CHECKING:
    from train_lora.train_lora_node import TrainLoraNode

logger = logging.getLogger("diffusers_nodes_library")


class FLUX1Parameters(TrainLoraModelFamilyParameters):
    def __init__(self, node: TrainLoraNode):
        self._node = node

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt1",
                default_value="FLUX.1",
                type="str",
                tooltip="The prompt or prompts to guide the image generation.",
            )
        )
        # TODO: Add all parameters

    def remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("prompt1")
        # TODO: Remove all parameters

    def get_script_kwargs(self) -> dict:
        kwargs = {
            "prompt": self._node.get_parameter_value("prompt1"),
            # TODO: Add all parameters
        }
        return kwargs
    
    def get_script_name(self) -> str:
        return "flux_train_network.py"
