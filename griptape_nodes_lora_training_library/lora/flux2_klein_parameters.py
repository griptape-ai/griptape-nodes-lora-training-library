from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from lora.model_family_parameters import TrainLoraModelFamilyParameters

if TYPE_CHECKING:
    from lora.train_lora_node import TrainLoraNode

logger = logging.getLogger("griptape_nodes_lora_training_library")


class FLUX2KleinParameters(TrainLoraModelFamilyParameters):
    PARAM_MODEL = "flux2_klein_model"

    def __init__(self, node: TrainLoraNode):
        self._node = node
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "black-forest-labs/FLUX.2-klein-base-4B",
                "black-forest-labs/FLUX.2-klein-4B",
            ],
            parameter_name=self.PARAM_MODEL,
        )
        self._dataset_config = Parameter(
            name="dataset_config_path",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="str",
            default_value="",
            tooltip="The full path to the dataset configuration file (.toml) or image directory.",
            traits={
                FileSystemPicker(
                    allow_files=True,
                    allow_directories=True,
                    multiple=False,
                    file_types=[".toml"],
                )
            },
        )
        self._output_dir = Parameter(
            name="output_dir",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
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
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="str",
            default_value="my_flux2_klein_lora",
            tooltip="The name of the output LoRA.",
        )
        self._learning_rate = Parameter(
            name="learning_rate",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="float",
            default_value=1e-4,
            tooltip="The learning rate for training.",
        )
        self._max_train_steps = Parameter(
            name="max_train_steps",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="int",
            default_value=1500,
            tooltip="Maximum number of training steps.",
        )
        self._save_every_n_steps = Parameter(
            name="save_every_n_steps",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="int",
            default_value=500,
            tooltip="Save a checkpoint every N steps (0 to disable).",
        )
        self._network_dim = Parameter(
            name="network_dim",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="int",
            default_value=16,
            tooltip="The dimension of the LoRA network (rank).",
        )
        self._network_alpha = Parameter(
            name="network_alpha",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="int",
            default_value=16,
            tooltip="The alpha parameter for the LoRA network.",
        )
        self._num_repeats = Parameter(
            name="num_repeats",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="int",
            default_value=10,
            tooltip="Number of times to repeat the dataset per epoch.",
        )
        self._resolution = Parameter(
            name="resolution",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="int",
            default_value=512,
            tooltip="Training resolution (images will be resized and center-cropped).",
        )
        self._mixed_precision = Parameter(
            name="mixed_precision",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="str",
            default_value="bf16",
            tooltip="Mixed precision training mode",
            traits={
                Options(choices=["bf16", "no"]),
            },
        )
        self._max_grad_norm = Parameter(
            name="max_grad_norm",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            type="float",
            default_value=1.0,
            tooltip="Maximum gradient norm for clipping.",
        )

        self._seed_parameter = SeedParameter(node)

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()
        self._node.add_parameter(self._dataset_config)
        self._node.add_parameter(self._output_dir)
        self._node.add_parameter(self._output_name)
        self._node.add_parameter(self._learning_rate)
        self._node.add_parameter(self._max_train_steps)
        self._node.add_parameter(self._save_every_n_steps)
        self._node.add_parameter(self._network_dim)
        self._node.add_parameter(self._network_alpha)
        self._node.add_parameter(self._num_repeats)
        self._node.add_parameter(self._resolution)
        self._node.add_parameter(self._mixed_precision)
        self._node.add_parameter(self._max_grad_norm)
        self._seed_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()
        self._node.remove_parameter_element_by_name(self._dataset_config.name)
        self._node.remove_parameter_element_by_name(self._output_dir.name)
        self._node.remove_parameter_element_by_name(self._output_name.name)
        self._node.remove_parameter_element_by_name(self._learning_rate.name)
        self._node.remove_parameter_element_by_name(self._max_train_steps.name)
        self._node.remove_parameter_element_by_name(self._save_every_n_steps.name)
        self._node.remove_parameter_element_by_name(self._network_dim.name)
        self._node.remove_parameter_element_by_name(self._network_alpha.name)
        self._node.remove_parameter_element_by_name(self._num_repeats.name)
        self._node.remove_parameter_element_by_name(self._resolution.name)
        self._node.remove_parameter_element_by_name(self._mixed_precision.name)
        self._node.remove_parameter_element_by_name(self._max_grad_norm.name)
        self._seed_parameter.remove_input_parameters()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._seed_parameter.after_value_set(parameter, value)

    def preprocess(self) -> None:
        self._seed_parameter.preprocess()

    def _resolve_dataset_path(self) -> str:
        """Resolve the image directory from dataset_config_path.

        If the path is a .toml file, parse it to extract image_dir.
        If it's a directory, use it directly.
        """
        import re

        config_path = Path(self._node.get_parameter_value("dataset_config_path"))
        if config_path.suffix == ".toml" and config_path.is_file():
            content = config_path.read_text()
            match = re.search(r"image_dir\s*=\s*['\"](.+?)['\"]", content)
            if match:
                return match.group(1)
            return str(config_path.parent)
        return str(config_path)

    def get_script_params(self) -> list[str]:
        repo_id = self._node.get_parameter_value(self.PARAM_MODEL)

        params = [
            "--pretrained_model_name_or_path",
            repo_id,
            "--dataset_path",
            self._resolve_dataset_path(),
            "--output_dir",
            self._node.get_parameter_value("output_dir"),
            "--output_name",
            self._node.get_parameter_value("output_name"),
            "--resolution",
            str(int(self._node.get_parameter_value("resolution"))),
            "--max_train_steps",
            str(int(self._node.get_parameter_value("max_train_steps"))),
            "--learning_rate",
            str(float(self._node.get_parameter_value("learning_rate"))),
            "--network_dim",
            str(int(self._node.get_parameter_value("network_dim"))),
            "--network_alpha",
            str(int(self._node.get_parameter_value("network_alpha"))),
            "--num_repeats",
            str(int(self._node.get_parameter_value("num_repeats"))),
            "--save_every_n_steps",
            str(int(self._node.get_parameter_value("save_every_n_steps"))),
            "--seed",
            str(int(self._seed_parameter.get_seed())),
            "--mixed_precision",
            self._node.get_parameter_value("mixed_precision"),
            "--max_grad_norm",
            str(float(self._node.get_parameter_value("max_grad_norm"))),
        ]

        return params

    def get_mixed_precision(self) -> str:
        return self._node.get_parameter_value("mixed_precision")

    def get_script_name(self) -> str:
        return "flux2_klein_train_lora.py"
