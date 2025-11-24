import logging
import subprocess
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from lora.train_lora_parameters import TrainLoraParameters

logger = logging.getLogger("griptape_nodes_lora_training_library")


class TrainLoraNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        self._initializing = True
        self.ui_options_cache: dict[str, dict] = {}
        super().__init__(**kwargs)

        self.params = TrainLoraParameters(self)
        self.params.add_input_parameters()

        self._create_status_parameters(
            result_details_tooltip="Details about the Lora training result",
            result_details_placeholder="Training result details will appear here.",
        )
        self._initializing = False

    def add_parameter(self, parameter: Parameter) -> None:
        """Add a parameter to the node.

        During initialization, parameters are added normally.
        After initialization (dynamic mode), parameters are marked as user-defined
        for serialization and duplicates are prevented.
        """
        if self._initializing:
            super().add_parameter(parameter)
            return

        # Dynamic mode: prevent duplicates and mark as user-defined
        if not self.does_name_exist(parameter.name):
            parameter.user_defined = True

            # Restore cached ui_options if available
            ui_options_to_restore = {"hide"}
            if parameter.name in self.ui_options_cache:
                parameter.ui_options = {
                    **parameter.ui_options,
                    **{k: v for k, v in self.ui_options_cache[parameter.name].items() if k in ui_options_to_restore},
                }

            super().add_parameter(parameter)

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        parameter = self.get_parameter_by_name(param_name)
        if parameter is None:
            return
        self.params.before_value_set(parameter, value)

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        self.params.after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        return self.params.validate_before_node_run()

    def save_ui_options(self) -> None:
        """Save ui_options for all current parameters to cache."""
        for element in self.root_ui_element.children:
            parameter = self.get_parameter_by_name(element.name)
            if parameter is not None and parameter.ui_options:
                self.ui_options_cache[parameter.name] = parameter.ui_options.copy()

    def clear_ui_options_cache(self) -> None:
        """Clear the ui_options cache."""
        self.ui_options_cache.clear()

    def _get_library_env_python(self) -> Path:
        # Check for Windows path
        python_exe_windows = Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe"
        if python_exe_windows.exists():
            logger.debug(f"Python executable found at: {python_exe_windows}")
            return python_exe_windows

        # Check for Unix-like (Mac/Linux) path
        python_exe_unix = Path(__file__).parent.parent / ".venv" / "bin" / "python"
        if python_exe_unix.exists():
            logger.debug(f"Python executable found at: {python_exe_unix}")
            return python_exe_unix

        raise FileNotFoundError(
            f"Python executable not found in expected locations: {python_exe_windows} or {python_exe_unix}"
        )

    def _generate_command(self, library_env_python: Path) -> list[str]:
        script_name = self.params.model_family_parameters.get_script_name()
        sd_scripts_dir = Path(__file__).parent.parent / "sd-scripts"
        script_path = sd_scripts_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        command = [
            str(library_env_python),
            "-u",
            "-m",
            "accelerate.commands.launch",
            "--num_cpu_threads_per_process",
            "1",
            "--mixed_precision",
            self.params.model_family_parameters.get_mixed_precision(),
            str(script_path),
        ]
        command.extend(self.params.model_family_parameters.get_script_params())
        logger.debug(f"Generated command: {command}")
        return command

    def preprocess(self) -> None:
        self._clear_execution_status()
        self.params.preprocess()

    def process(self) -> None:
        self.preprocess()
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
            subprocess.run(args=command)

            # Set the lora_path output parameter
            output_dir = self.get_parameter_value("output_dir")
            output_name = self.get_parameter_value("output_name")
            lora_path = Path(output_dir) / f"{output_name}.safetensors"
            self.set_parameter_value("lora_path", str(lora_path))

            success_msg = f"Lora training executed successfully."
            self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_msg}")
        except Exception as e:
            error_msg = f"Failed to execute lora training: {e}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            self._handle_failure_exception(e)
