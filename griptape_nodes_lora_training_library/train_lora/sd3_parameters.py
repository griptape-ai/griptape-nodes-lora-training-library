from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.common.parameters.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from train_lora.model_family_parameters import TrainLoraModelFamilyParameters

if TYPE_CHECKING:
    from train_lora.train_lora_node import TrainLoraNode

logger = logging.getLogger("diffusers_nodes_library")


class SD3Parameters(TrainLoraModelFamilyParameters):
    def __init__(self, node: TrainLoraNode):
        self._node = node
        self._huggingface_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "stabilityai/stable-diffusion-3.5-medium",
                "stabilityai/stable-diffusion-3.5-large",
                "stabilityai/stable-diffusion-3.5-large-turbo",
                "stabilityai/stable-diffusion-3-medium-diffusers",
            ],
            parameter_name="sd3_model",
        )

    def add_input_parameters(self) -> None:
        self._huggingface_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._huggingface_repo_parameter.remove_input_parameters()

    def get_script_params(self) -> list[str]:
        return []

    def get_mixed_precision(self) -> str:
        return "bf16"

    def get_script_name(self) -> str:
        return "sd3_train_network.py"
