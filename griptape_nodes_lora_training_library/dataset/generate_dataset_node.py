import logging
from pathlib import Path


from schema import Literal, Schema

from griptape.artifacts import ImageArtifact, ImageUrlArtifact, ListArtifact
from griptape.drivers.prompt.griptape_cloud_prompt_driver import GriptapeCloudPromptDriver
from griptape.engines import JsonExtractionEngine
from griptape.structures import Agent

from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options

from error_utils import try_throw_error
from image_utils import load_image_from_url_artifact

logger = logging.getLogger("griptape_nodes_lora_training_library")


API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
RESOLUTION_OPTIONS = [512, 1024]

class GenerateDatasetNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="agent",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="Agent",
                output_type="Agent",
                tooltip="The agent that will be used to describe the images. Defaults to a GPT-4.1-mini agent if not provided.",
                default_value=None,
            )
        )

        self.add_parameter(
            Parameter(
                name="agent_prompt",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value = "Describe this image with descriptive tags. Include details about the subject, setting, colors, mood, and style.",
                tooltip="The prompt to use for the agent when describing images.",
                ui_options={
                    "multiline": True,
                },
            )
        )

        self.add_parameter(
            ParameterList(
                name="images",
                input_types=["ImageUrlArtifact", "ImageArtifact"],
                default_value=[],
                allowed_modes={ParameterMode.INPUT},
                tooltip="Images to include in the dataset.",
            )
        )

        self.add_parameter(
            Parameter(
                name="image_resolution",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=RESOLUTION_OPTIONS[1],
                tooltip="The resolution of the images in the dataset.",
                traits={
                    Options(
                        choices=RESOLUTION_OPTIONS
                    )
                }
            )
        )

        self.add_parameter(
            Parameter(
                name="num_repeats",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=8,
                tooltip="The number of times to repeat the dataset during training.",
            )
        )

        self.add_parameter(
            Parameter(
                name="dataset_folder",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="The full path where the generated dataset directory will be created.",
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
                name="dataset_config_path",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="str",
                default_value="<dataset_config_path>",
                tooltip="The full path to the dataset configuration file.",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the dataset generation result",
            result_details_placeholder="Dataset generation result details will appear here.",
        )

    def create_dataset(self, dataset_folder: Path, images: list[ImageArtifact | ImageUrlArtifact]):
        dataset_folder.mkdir(parents=True, exist_ok=True)

        images_folder = dataset_folder / "images"
        images_folder.mkdir(parents=True, exist_ok=True)

        agent = self.get_parameter_value("agent")
        if not agent:
            prompt_driver = GriptapeCloudPromptDriver(
                model="gpt-4.1-mini",
                api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
                stream=False,
            )
            agent = Agent(prompt_driver=prompt_driver)

        # Create a JSON extraction engine to extract structured tags
        tag_schema = Schema({
            Literal("tags", description="List of descriptive tags for the image"): [str]
        })
        extraction_engine = JsonExtractionEngine(
            prompt_driver=agent.prompt_driver,
            template_schema=tag_schema.json_schema("TagSchema")
        )

        for i, image_artifact in enumerate(images):
            # Convert ImageUrlArtifact to ImageArtifact if needed
            if isinstance(image_artifact, ImageUrlArtifact):
                image_artifact = load_image_from_url_artifact(image_artifact)

            # Generate a filename for the image using the artifact's format
            image_filename = f"image_{i:04d}.{image_artifact.format}"
            image_path = images_folder / image_filename

            # Save the image artifact to disk
            with open(image_path, 'wb') as f:
                f.write(image_artifact.to_bytes())
            logger.info(f"Saved image to {image_path}")

            # Use the agent to generate descriptive tags for the image
            prompt = f"{self.get_parameter_value("agent_prompt")} The output must be a single JSON object with a 'tags' field containing a list of tags."
            agent.run([prompt, image_artifact])
            try_throw_error(agent.output)
            extraction_result = extraction_engine.extract_artifacts(ListArtifact([agent.output]))

            # Parse the extracted JSON to get tags
            tags = extraction_result[0].value["tags"]
            caption_text = ", ".join(tags)
            logger.info(f"Generated {len(tags)} tags for {image_filename}: {caption_text}")

            # Write the caption file directly to images folder (FLUX style)
            base_name = Path(image_filename).stem
            caption_filename = f"{base_name}.txt"
            (images_folder / caption_filename).write_text(caption_text)

        return
    
    def generate_toml(self, output_folder: Path, resolution: int, num_repeats: int) -> Path:
        images_folder = output_folder / "images"
        toml = f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 0

[[datasets]]
resolution = {resolution}
batch_size = 1
keep_tokens = 0

  [[datasets.subsets]]
  image_dir = '{images_folder}'
  num_repeats = {num_repeats}"""

        toml_path = output_folder / "dataset.toml"
        toml_path.write_text(toml)
        return toml_path

    def process(self) -> None:
        self._clear_execution_status()
        dataset_folder = self.get_parameter_value("dataset_folder")
        image_resolution = self.get_parameter_value("image_resolution")
        num_repeats = self.get_parameter_value("num_repeats")
        images = self.get_parameter_value("images")

        if not dataset_folder:
            self._set_status_results(was_successful=False, result_details="Dataset folder path is required.")
            return

        dataset_folder_path = Path(dataset_folder)

        if not images or len(images) == 0:
            self._set_status_results(was_successful=False, result_details="No images provided. Please connect images to the images input.")
            return

        logger.info(f"Processing {len(images)} images")

        try:
            self.create_dataset(
                dataset_folder_path,
                images
            )
            logger.info(f"Dataset created at: {dataset_folder_path}")
        except Exception as e:
            logger.exception(f"Error while creating dataset: {str(e)}")
            self._set_status_results(was_successful=False, result_details=f"Error while creating dataset: {str(e)}")
            return

        try:
            dataset_toml_path = self.generate_toml(
                output_folder=dataset_folder_path,
                resolution=image_resolution,
                num_repeats=num_repeats
            )
            logger.info(f"Dataset configuration file generated at: {dataset_toml_path}")
        except Exception as e:
            logger.exception(f"Error while generating dataset.toml: {str(e)}")
            self._set_status_results(was_successful=False, result_details=f"Error while generating dataset.toml: {str(e)}")
            return

        self.set_parameter_value("dataset_config_path", str(dataset_toml_path))
        self._set_status_results(was_successful=True, result_details="Success")
