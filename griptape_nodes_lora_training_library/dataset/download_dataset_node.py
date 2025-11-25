import logging
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.traits.file_system_picker import FileSystemPicker

logger = logging.getLogger("griptape_nodes_lora_training_library")


class DownloadDatasetNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="url",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="URL to download the zip file from.",
            )
        )

        self.add_parameter(
            Parameter(
                name="extract_location",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="The full path where the zip file will be extracted.",
                traits={
                    FileSystemPicker(
                        allow_files=False,
                        allow_directories=True,
                        multiple=False,
                    )
                },
            )
        )

        self.add_parameter(
            Parameter(
                name="extracted_path",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="str",
                default_value="<extracted_path>",
                tooltip="The full path to the extracted dataset.",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the download and extraction result",
            result_details_placeholder="Download result details will appear here.",
        )

    def _process(self) -> None:
        self._clear_execution_status()
        url = self.get_parameter_value("url")
        extract_location = self.get_parameter_value("extract_location")

        if not url:
            self._set_status_results(was_successful=False, result_details="URL is required.")
            return

        if not extract_location:
            self._set_status_results(was_successful=False, result_details="Extract location is required.")
            return

        extract_path = Path(extract_location)

        try:
            # Create the extract location if it doesn't exist
            extract_path.mkdir(parents=True, exist_ok=True)
            msg = f"Extract location verified at: {extract_path}"
            self.status_component.append_to_result_details(msg)

            # Download the zip file
            msg = f"Downloading zip file from: {url}"
            self.status_component.append_to_result_details(msg)
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Get filename from URL or use default
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name or "dataset.zip"
            if not filename.endswith(".zip"):
                filename += ".zip"

            zip_path = extract_path / filename

            # Save the downloaded file
            with zip_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            msg = f"Downloaded zip file to: {zip_path}"
            self.status_component.append_to_result_details(msg)

            # Extract the zip file
            msg = f"Extracting zip file to: {extract_path}"
            self.status_component.append_to_result_details(msg)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            msg = f"Extraction complete. Files extracted to: {extract_path}"
            self.status_component.append_to_result_details(msg)

            # Optionally remove the zip file after extraction
            zip_path.unlink()
            msg = f"Removed zip file: {zip_path}"
            self.status_component.append_to_result_details(msg)

            # Check if a folder with the same name as the zip file was created
            dataset_folder_name = Path(filename).stem
            potential_folder = extract_path / dataset_folder_name
            if potential_folder.exists() and potential_folder.is_dir():
                # Use the folder with the zip file's name as the extracted path
                final_extract_path = potential_folder
                msg = f"Detected folder matching zip name: {final_extract_path}"
                self.status_component.append_to_result_details(msg)
            else:
                final_extract_path = extract_path

            self.set_parameter_value("extracted_path", str(final_extract_path))
            self.publish_update_to_parameter("extracted_path", str(final_extract_path))
            self._set_status_results(
                was_successful=True,
                result_details=f"Successfully downloaded and extracted dataset to {final_extract_path}",
            )

        except requests.exceptions.RequestException as e:
            msg = f"Error while downloading file: {e!s}"
            logger.exception(msg)
            self._set_status_results(was_successful=False, result_details=msg)
            return
        except zipfile.BadZipFile as e:
            msg = f"Error: Invalid zip file: {e!s}"
            logger.exception(msg)
            self._set_status_results(was_successful=False, result_details=msg)
            return
        except Exception as e:
            msg = f"Error while extracting zip file: {e!s}"
            logger.exception(msg)
            self._set_status_results(was_successful=False, result_details=msg)
            return

    def process(
        self,
    ) -> AsyncResult[None]:
        yield lambda: self._process()
