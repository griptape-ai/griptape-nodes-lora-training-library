from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from lora.model_family_parameters import TrainLoraModelFamilyParameters

if TYPE_CHECKING:
    from pathlib import Path

    from lora.train_lora_node import TrainLoraNode

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
        self._dataset_config = Parameter(
            name="dataset_config_path",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="str",
            default_value="",
            tooltip="The full path to the dataset configuration file.",
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
            tooltip="The full path to the output directory.",
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
            tooltip="The name of the output LoRA.",
        )
        self._learning_rate = Parameter(
            name="learning_rate",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="float",
            default_value=1e-4,
            tooltip="The learning rate for training.",
        )
        self._save_every_n_epochs = Parameter(
            name="save_every_n_epochs",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="int",
            default_value=1,
            tooltip="Save the model every N epochs.",
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
            default_value=1.0,
            tooltip="The alpha parameter for the network.",
        )
        self._full_bf16 = Parameter(
            name="full_bf16",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="bool",
            default_value=False,
            tooltip="Whether to use full bf16 precision.",
        )
        self._mixed_precision = Parameter(
            name="mixed_precision",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="str",
            default_value="bf16",
            tooltip="Mixed precision training mode",
            traits={
                Options(choices=["bf16", "fp16", "no"]),
            },
        )
        self._save_precision = Parameter(
            name="save_precision",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            type="str",
            default_value="bf16",
            tooltip="Precision for saving models",
            traits={
                Options(choices=["bf16", "fp16", "no"]),
            },
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
        self._highvram = Parameter(
            name="highvram",
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
        self._node.add_parameter(self._dataset_config)
        self._node.add_parameter(self._output_dir)
        self._node.add_parameter(self._output_name)
        self._node.add_parameter(self._learning_rate)
        self._node.add_parameter(self._save_every_n_epochs)
        self._node.add_parameter(self._max_train_epochs)
        self._node.add_parameter(self._network_dim)
        self._node.add_parameter(self._network_alpha)
        self._node.add_parameter(self._full_bf16)
        self._node.add_parameter(self._mixed_precision)
        self._node.add_parameter(self._save_precision)
        self._node.add_parameter(self._guidance_scale)
        self._node.add_parameter(self._fp8_base)
        self._node.add_parameter(self._highvram)
        self._node.add_parameter(self._max_data_loader_n_workers)
        self._seed_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()
        self._node.remove_parameter_element_by_name(self._dataset_config.name)
        self._node.remove_parameter_element_by_name(self._output_dir.name)
        self._node.remove_parameter_element_by_name(self._output_name.name)
        self._node.remove_parameter_element_by_name(self._learning_rate.name)
        self._node.remove_parameter_element_by_name(self._save_every_n_epochs.name)
        self._node.remove_parameter_element_by_name(self._max_train_epochs.name)
        self._node.remove_parameter_element_by_name(self._network_dim.name)
        self._node.remove_parameter_element_by_name(self._network_alpha.name)
        self._node.remove_parameter_element_by_name(self._full_bf16.name)
        self._node.remove_parameter_element_by_name(self._mixed_precision.name)
        self._node.remove_parameter_element_by_name(self._save_precision.name)
        self._node.remove_parameter_element_by_name(self._guidance_scale.name)
        self._node.remove_parameter_element_by_name(self._fp8_base.name)
        self._node.remove_parameter_element_by_name(self._highvram.name)
        self._node.remove_parameter_element_by_name(self._max_data_loader_n_workers.name)
        self._seed_parameter.remove_input_parameters()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._seed_parameter.after_value_set(parameter, value)

    def preprocess(self) -> None:
        self._seed_parameter.preprocess()

    def _get_flux_model_path(self) -> Path:
        flux_patterns = [
            "flux*.safetensors",  # Any flux model file
            "diffusion_pytorch_model.safetensors",  # Standard diffusers format
        ]
        flux_model_path = self.get_model_file_path(flux_patterns, self._model_repo_parameter)
        logger.debug(f"Using FLUX model file at: {flux_model_path}")
        return flux_model_path

    def _get_ae_model_path(self) -> Path:
        ae_patterns = [
            "ae.safetensors",
            "ae.sft",
            "vae/*.safetensors",
        ]
        ae_path = self.get_model_file_path(ae_patterns, self._model_repo_parameter)
        logger.debug(f"Using AE model file at: {ae_path}")
        return ae_path

    def _get_clip_l_model_path(self) -> Path:
        clip_l_patterns = [
            "text_encoder/model.safetensors",
            "text_encoder/model-00001-of-*.safetensors",  # Sharded model (will be merged)
        ]
        clip_l_path = self.get_model_file_path(clip_l_patterns, self._model_repo_parameter)
        logger.debug(f"Using CLIP-L model file at: {clip_l_path}")
        return clip_l_path

    def _get_t5xxl_model_path(self) -> Path:
        t5xxl_patterns = [
            "text_encoder_2/model.safetensors",
            "text_encoder_2/model-00001-of-*.safetensors",  # Sharded model (will be merged)
        ]
        t5xxl_path = self.get_model_file_path(t5xxl_patterns, self._model_repo_parameter)
        logger.debug(f"Using T5XXL model file at: {t5xxl_path}")
        return t5xxl_path

    def get_script_params(self) -> list[str]:
        def format_float(value: float) -> str:
            """Format a float to always include at least one decimal place."""
            formatted = str(float(value))
            if "." not in formatted and "e" not in formatted.lower():
                formatted += ".0"
            return formatted

        hardcoded_params = [
            "--cache_latents_to_disk",
            "--sdpa",
            "--persistent_data_loader_workers",
            "--gradient_checkpointing",
            "--cache_text_encoder_outputs",
            "--discrete_flow_shift",
            "3.1582",
            "--timestep_sampling",
            "shift",
            "--network_module",
            "networks.lora_flux",
            "--save_model_as",
            "safetensors",
            "--loss_type",
            "l2",
            "--model_prediction_type",
            "raw",
        ]

        key_value_params = [
            # Model file paths resolved from HuggingFace cache
            "--pretrained_model_name_or_path",
            str(self._get_flux_model_path()),
            "--clip_l",
            str(self._get_clip_l_model_path()),
            "--t5xxl",
            str(self._get_t5xxl_model_path()),
            "--ae",
            str(self._get_ae_model_path()),
            # Training configuration
            "--dataset_config",
            self._node.get_parameter_value("dataset_config_path"),
            "--output_dir",
            self._node.get_parameter_value("output_dir"),
            "--output_name",
            self._node.get_parameter_value("output_name"),
            "--learning_rate",
            format_float(self._node.get_parameter_value("learning_rate")),
            "--save_every_n_epochs",
            str(int(self._node.get_parameter_value("save_every_n_epochs"))),
            "--max_train_epochs",
            str(int(self._node.get_parameter_value("max_train_epochs"))),
            "--network_dim",
            str(int(self._node.get_parameter_value("network_dim"))),
            "--network_alpha",
            format_float(self._node.get_parameter_value("network_alpha")),
            "--mixed_precision",
            self._node.get_parameter_value("mixed_precision"),
            "--save_precision",
            self._node.get_parameter_value("save_precision"),
            "--guidance_scale",
            format_float(self._node.get_parameter_value("guidance_scale")),
            "--max_data_loader_n_workers",
            str(int(self._node.get_parameter_value("max_data_loader_n_workers"))),
            "--seed",
            str(int(self._seed_parameter.get_seed())),
        ]
        params = hardcoded_params + key_value_params
        if self._node.get_parameter_value("full_bf16"):
            params.append("--full_bf16")
        if self._node.get_parameter_value("fp8_base"):
            params.append("--fp8_base")
        if self._node.get_parameter_value("highvram"):
            params.append("--highvram")
        return params

    def get_mixed_precision(self) -> str:
        return self._node.get_parameter_value("mixed_precision")

    def get_script_name(self) -> str:
        return "flux_train_network.py"
