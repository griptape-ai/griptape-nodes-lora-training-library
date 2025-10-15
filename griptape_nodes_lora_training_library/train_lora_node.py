import logging
from pathlib import Path
import subprocess
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes_lora_training_library")


MODEL_FAMILIES = ["FLUX.1", "SD3", "SDXL"]

class TrainLora(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="model_family",
                type="str",
                tooltip="The model family to use for training",
                default_value=MODEL_FAMILIES[0],
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_FAMILIES)},
            )
        )
        # TODO: Dynamically show rest of the parameters based on model choice

        self._create_status_parameters(
            result_details_tooltip="Details about the Lora training result",
            result_details_placeholder="Training result details will appear here.",
        )

    def _get_library_env_python(self) -> Path:
        python_exe = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
        if python_exe.exists():
            logger.info(f"Python executable found at: {python_exe}")
            return python_exe
        else:
            raise FileNotFoundError("Python executable not found in the expected location.")

    def _generate_command(self, library_env_python: Path) -> list[str]:
        # TODO: Implement actual command generation logic based on parameters
        command = [str(library_env_python), "-c", "print('Hello World')"]
        logger.info(f"Generated command: {command}")
        return command

    def process(self) -> None:
        self._clear_execution_status()

        logger.warning("Starting Lora training process...")

        try:
            library_env_python = self._get_library_env_python()
        except Exception as e:
            error_msg = f"Failed to find python executable: {e}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            self._handle_failure_exception(e)
            return
            
        try:
            command = self._generate_command(library_env_python)
        except Exception as e:
            error_msg = f"Failed to generate lora training command: {e}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            self._handle_failure_exception(e)
            return
            
        try:
            subprocess.run(command)
            success_msg = f"Lora training executed successfully."
            self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_msg}")
        except Exception as e:
            error_msg = f"Failed to execute lora training: {e}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            self._handle_failure_exception(e)