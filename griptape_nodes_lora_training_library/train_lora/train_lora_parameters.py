from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar
from abc import ABC

from train_lora.model_family_parameters import TrainLoraModelFamilyParameters
from train_lora.flux1_parameters import FLUX1Parameters

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options

if TYPE_CHECKING:
    from train_lora.train_lora_node import TrainLoraNode

logger = logging.getLogger("griptape_nodes_lora_training_library")


MODEL_FAMILIES = ["FLUX.1"]

class TrainLoraParameters(ABC):
    START_PARAMS: ClassVar = ["lora", "model_family"]
    END_PARAMS: ClassVar = ["Status"]
    def __init__(self, node: TrainLoraNode):
        self._node = node
        self._model_family_parameters: TrainLoraModelFamilyParameters
        self.did_model_family_change = False
        self.set_model_family_parameters(MODEL_FAMILIES[0])

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="model_family",
                type="str",
                tooltip="The model family to use for training",
                default_value=MODEL_FAMILIES[0],
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_FAMILIES)},
            )
        )
        self.model_family_parameters.add_input_parameters()

    def add_output_parameters(self) -> None:
        # TODO: Make this a Lora output
        self._node.add_parameter(
            Parameter(
                name="lora",
                output_type="Lora",
                default_value=None,
                tooltip="Trained Lora",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "lora"},
            )
        )

    def set_model_family_parameters(self, model_family: str) -> None:
        match model_family:
            case "FLUX.1":
                self._model_family_parameters = FLUX1Parameters(self._node)
            case _:
                msg = f"Unsupported model family: {model_family}"
                logger.error(msg)
                raise ValueError(msg)

    def before_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "model_family":
            current_model_family = self._node.get_parameter_value("model_family")
            self.did_model_family_change = current_model_family != value

    def after_value_set(self, parameter: Parameter, _value: Any) -> None:
        if parameter.name == "model_family" and self.did_model_family_change:
            self.regenerate_elements_for_model_family()

    def regenerate_elements_for_model_family(self) -> None:
        self._node.save_ui_options()

        self.model_family_parameters.remove_input_parameters()
        self.set_model_family_parameters(self.get_model_family())
        self.model_family_parameters.add_input_parameters()

        # Get all current element names
        all_element_names = [element.name for element in self._node.root_ui_element.children]

        # Build parameter groupings
        start_params = TrainLoraParameters.START_PARAMS
        end_params = TrainLoraParameters.END_PARAMS
        excluded_params = {*start_params, *end_params}

        # Assemble final order: start -> middle -> end
        middle_params = [name for name in all_element_names if name not in excluded_params]
        sorted_parameters = [*start_params, *middle_params, *end_params]

        self._node.reorder_elements(sorted_parameters)

        self._node.clear_ui_options_cache()

    @property
    def model_family_parameters(self) -> TrainLoraModelFamilyParameters:
        if self._model_family_parameters is None:
            msg = "Model family parameters not initialized. Ensure model family parameter is set."
            logger.error(msg)
            raise ValueError(msg)
        return self._model_family_parameters

    def get_model_family(self) -> str:
        return self._node.get_parameter_value("model_family")
    
    def validate_before_node_run(self) -> list[Exception] | None:
        return self.model_family_parameters.validate_before_node_run()
