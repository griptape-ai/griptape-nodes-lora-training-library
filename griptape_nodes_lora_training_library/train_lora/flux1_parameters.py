from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.common.parameters.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from train_lora.model_family_parameters import TrainLoraModelFamilyParameters

if TYPE_CHECKING:
    from train_lora.train_lora_node import TrainLoraNode

logger = logging.getLogger("diffusers_nodes_library")


class FLUX1Parameters(TrainLoraModelFamilyParameters):
    def __init__(self, node: TrainLoraNode):
        self._node = node
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "black-forest-labs/FLUX.1-schnell",
                "black-forest-labs/FLUX.1-dev",
                "black-forest-labs/FLUX.1-Krea-dev",
            ],
            parameter_name="flux_model",
        )

        self._text_encoder_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "openai/clip-vit-large-patch14",
            ],
            parameter_name="text_encoder",
        )

        self._text_encoder_2_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "google/t5-v1_1-xxl",
            ],
            parameter_name="text_encoder_2",
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()
        self._text_encoder_repo_parameter.add_input_parameters()
        self._text_encoder_2_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()
        self._text_encoder_repo_parameter.remove_input_parameters()
        self._text_encoder_2_repo_parameter.remove_input_parameters()

    def get_script_kwargs(self) -> dict:
        kwargs = {
            "flux_model": self._node.get_parameter_value("flux_model"),
            "text_encoder": self._node.get_parameter_value("text_encoder"),
            "text_encoder_2": self._node.get_parameter_value("text_encoder_2"),
            # TODO: Add all parameters
        }
        return kwargs
    
    def get_script_name(self) -> str:
        return "flux_train_network.py"
