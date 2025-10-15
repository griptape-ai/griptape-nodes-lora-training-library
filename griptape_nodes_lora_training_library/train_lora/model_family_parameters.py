from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from train_lora.train_lora_node import TrainLoraNode

logger = logging.getLogger("diffusers_nodes_library")


class TrainLoraModelFamilyParameters(ABC):
    def __init__(self, node: TrainLoraNode):
        self._node = node

    def validate_before_node_run(self) -> list[Exception] | None:
        return None

    @abstractmethod
    def add_input_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_input_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_script_kwargs(self) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def get_script_name(self) -> str:
        raise NotImplementedError
