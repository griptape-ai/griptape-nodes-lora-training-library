from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from griptape_nodes.common.parameters.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.common.parameters.seed_parameter import SeedParameter
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options

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
        self._dataset_config = Parameter(
            name="dataset_config",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="str",
            default_value="",
            tooltip="The full path to the loaded file.",
            traits={
                    FileSystemPicker(
                        allow_files=True,
                        allow_directories=False,
                        multiple=False,
                        file_types=[
                            ".toml",
                        ],
                    )   
                },
        )
        self._output_dir = Parameter(
            name="output_dir",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="str",
            default_value="",
            tooltip="The full path to the loaded file.",
            traits={
                    FileSystemPicker(
                        allow_files=False,
                        allow_directories=True,
                        multiple=False,
                    )   
                },
        )
        self._output_name = Parameter(
            name="output_name",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="str",
            default_value="my_flux_lora",
            tooltip="The name of the output Lora.",
        )
        self._learning_rate = Parameter(
            name="learning_rate",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="float",
            default_value=1e-6,
            tooltip="The learning rate for training.",
        )
        self._max_train_epochs = Parameter(
            name="max_train_epochs",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="int",
            default_value=10,
            tooltip="The maximum number of training epochs.",
        )
        self._network_dim = Parameter(
            name="network_dim",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="int",
            default_value=4,
            tooltip="The dimension of the network.",
        )
        self._network_alpha = Parameter(
            name="network_alpha",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="float",
            default_value=1e-3,
            tooltip="The alpha parameter for the network.",
        )
        self._mixed_precision = Parameter(
            name="mixed_precision",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="str",
            default_value="bf16",
            tooltip="Mixed precision training mode",
            traits={
                Options(choices=["bf16", "fp16", "no"]),
            }
        )
        self._save_precision = Parameter(
            name="save_precision",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="str",
            default_value="bf16",
            tooltip="Precision for saving models",
            traits={
                Options(choices=["bf16", "fp16", "no"]),
            }
        )
        self._guidance_scale = Parameter(
            name="guidance_scale",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="float",
            default_value=1.0,
            tooltip="The guidance scale to use during training.",
        )
        self._fp8_base = Parameter(
            name="fp8_base",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="bool",
            default_value=True,
            tooltip="Whether to quantize models to fp8.",
        )
        self._high_vram = Parameter(
            name="high_vram",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="bool",
            default_value=True,
            tooltip="Whether to use high VRAM mode.",
        )
        self._max_data_loader_n_workers = Parameter(
            name="max_data_loader_n_workers",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="int",
            default_value=1,
            tooltip="The maximum number of data loader workers.",
        )

        self._seed_parameter = SeedParameter(node)
        

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()
        self._text_encoder_repo_parameter.add_input_parameters()
        self._text_encoder_2_repo_parameter.add_input_parameters()
        self._node.add_parameter(self._dataset_config)
        self._node.add_parameter(self._output_dir)
        self._node.add_parameter(self._output_name)
        self._node.add_parameter(self._learning_rate)
        self._node.add_parameter(self._max_train_epochs)
        self._node.add_parameter(self._network_dim)
        self._node.add_parameter(self._network_alpha)
        self._node.add_parameter(self._mixed_precision)
        self._node.add_parameter(self._save_precision)
        self._node.add_parameter(self._guidance_scale)
        self._node.add_parameter(self._fp8_base)
        self._node.add_parameter(self._high_vram)
        self._node.add_parameter(self._max_data_loader_n_workers)
        self._seed_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()
        self._text_encoder_repo_parameter.remove_input_parameters()
        self._text_encoder_2_repo_parameter.remove_input_parameters()
        self._node.remove_parameter_by_name(self._dataset_config.name)
        self._node.remove_parameter_by_name(self._output_dir.name)
        self._node.remove_parameter_by_name(self._output_name.name)
        self._node.remove_parameter_by_name(self._learning_rate.name)
        self._node.remove_parameter_by_name(self._max_train_epochs.name)
        self._node.remove_parameter_by_name(self._network_dim.name)
        self._node.remove_parameter_by_name(self._network_alpha.name)
        self._node.remove_parameter_by_name(self._mixed_precision.name)
        self._node.remove_parameter_by_name(self._save_precision.name)
        self._node.remove_parameter_by_name(self._guidance_scale.name)
        self._node.remove_parameter_by_name(self._fp8_base.name)
        self._node.remove_parameter_by_name(self._high_vram.name)
        self._node.remove_parameter_by_name(self._max_data_loader_n_workers.name)
        self._seed_parameter.remove_input_parameters()

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
