import logging
from pathlib import Path
from typing import Any

from error_utils import try_throw_error
from griptape.artifacts import ImageArtifact, ImageUrlArtifact, ListArtifact
from griptape.drivers.prompt.griptape_cloud_prompt_driver import GriptapeCloudPromptDriver
from griptape.engines import JsonExtractionEngine
from griptape.loaders import ImageLoader
from griptape.structures import Agent
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from image_utils import load_image_from_url_artifact
from schema import Literal, Schema

logger = logging.getLogger("griptape_nodes_lora_training_library")


API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
RESOLUTION_OPTIONS = [512, 1024]


class GenerateDatasetNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="images",
                input_types=["list"],
                default_value=[],
                allowed_modes={ParameterMode.INPUT},
                tooltip="Images to include in the dataset.",
            )
        )

        self.add_parameter(
            Parameter(
                name="generate_captions",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="bool",
                default_value=True,
                tooltip="Whether to generate captions for the images using the agent.",
            )
        )

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
                default_value="Describe this image with descriptive tags. Include details about the subject, setting, colors, mood, and style.",
                tooltip="The prompt to use for the agent when describing images.",
                ui_options={
                    "multiline": True,
                },
            )
        )

        self.add_parameter(
            Parameter(
                name="captions",
                input_types=["list"],
                default_value=[],
                allowed_modes={ParameterMode.INPUT},
                tooltip="Captions to include in the dataset.",
            )
        )

        self.add_parameter(
            Parameter(
                name="trigger_phrase",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="Optional trigger phrase to prepend to all captions. Leave empty to not use a trigger phrase.",
            )
        )

        self.add_parameter(
            Parameter(
                name="image_resolution",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=RESOLUTION_OPTIONS[1],
                tooltip="The resolution of the images in the dataset.",
                traits={Options(choices=RESOLUTION_OPTIONS)},
            )
        )

        self.add_parameter(
            Parameter(
                name="num_repeats",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=1,
                tooltip="The number of times to repeat the dataset during training.",
                ui_options={
                    "hide": True,
                },
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

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "generate_captions":
            if value:
                self.show_parameter_by_name("agent")
                self.show_parameter_by_name("agent_prompt")
                self.hide_parameter_by_name("captions")
            else:
                self.hide_parameter_by_name("agent")
                self.hide_parameter_by_name("agent_prompt")
                self.show_parameter_by_name("captions")

    def _generate_caption_for_image(
        self, image_artifact: ImageArtifact, agent: Agent, extraction_engine: JsonExtractionEngine
    ) -> str:
        # Use the agent to generate descriptive tags for the image
        prompt = f"{self.get_parameter_value('agent_prompt')} The output must be a single JSON object with a 'tags' field containing a list of tags."
        agent.run([prompt, image_artifact])
        try_throw_error(agent.output)
        extraction_result = extraction_engine.extract_artifacts(ListArtifact([agent.output]))

        # Parse the extracted JSON to get tags
        tags = extraction_result[0].value["tags"]
        caption_text = ", ".join(tags)
        logger.debug(f"Generated {len(tags)} tags for {image_artifact}: {caption_text}")
        return caption_text

    def create_dataset(self, dataset_folder: Path, images: list[ImageArtifact | ImageUrlArtifact] | None):
        dataset_folder.mkdir(parents=True, exist_ok=True)

        images_folder = dataset_folder / "images"
        images_folder.mkdir(parents=True, exist_ok=True)

        # If no images provided, scan the dataset_folder for existing images
        if not images or len(images) == 0:
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
            existing_image_files = [
                f for f in dataset_folder.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
            ]

            if not existing_image_files:
                raise ValueError(f"No images found in dataset folder: {dataset_folder}")

            logger.debug(f"Found {len(existing_image_files)} existing images in dataset folder")

            images = []
            image_loader = ImageLoader()
            for image_file in existing_image_files:
                with open(image_file, "rb") as f:
                    image_data = f.read()

                # Parse image using Griptape's ImageLoader
                image_artifact = image_loader.parse(image_data)
                image_artifact.name = image_file.name
                images.append(image_artifact)

        if self.get_parameter_value("generate_captions"):
            agent = self.get_parameter_value("agent")
            if not agent:
                prompt_driver = GriptapeCloudPromptDriver(
                    model="gpt-4.1-mini",
                    api_key=GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR),
                    stream=False,
                )
                agent = Agent(prompt_driver=prompt_driver)

            # Create a JSON extraction engine to extract structured tags
            tag_schema = Schema({Literal("tags", description="List of descriptive tags for the image"): [str]})
            extraction_engine = JsonExtractionEngine(
                prompt_driver=agent.prompt_driver, template_schema=tag_schema.json_schema("TagSchema")
            )

        for i, image_artifact in enumerate(images):
            # Convert ImageUrlArtifact to ImageArtifact if needed
            if isinstance(image_artifact, ImageUrlArtifact):
                image_artifact = load_image_from_url_artifact(image_artifact)

            # Use existing filename if available, otherwise generate one
            if hasattr(image_artifact, "name") and image_artifact.name:
                image_filename = image_artifact.name
            else:
                image_filename = f"image_{i:04d}.{image_artifact.format}"

            image_path = images_folder / image_filename

            # Only save if image doesn't already exist in images folder
            source_path = dataset_folder / image_filename
            if source_path.exists() and source_path != image_path:
                # Copy existing image from dataset_folder to images_folder
                import shutil

                shutil.copy2(str(source_path), str(image_path))
                logger.debug(f"Copied existing image to {image_path}")
            elif not image_path.exists():
                # Save the image artifact to disk
                with open(image_path, "wb") as f:
                    f.write(image_artifact.to_bytes())
                logger.debug(f"Saved image to {image_path}")

            if self.get_parameter_value("generate_captions"):
                caption_text = self._generate_caption_for_image(image_artifact, agent, extraction_engine)
            else:
                caption_text = self.get_parameter_value("captions")[i]

            # Prepend trigger phrase if provided
            trigger_phrase = self.get_parameter_value("trigger_phrase")
            if trigger_phrase and trigger_phrase.strip():
                caption_text = f"{trigger_phrase.strip()}, {caption_text}"

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

        # Check if images exist in dataset_folder when no images are provided via input
        if not images or len(images) == 0:
            image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
            existing_images = (
                [f for f in dataset_folder_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
                if dataset_folder_path.exists()
                else []
            )

            if not existing_images:
                self._set_status_results(
                    was_successful=False,
                    result_details="No images provided via input and no images found in dataset folder. Please connect images to the images input or place images in the dataset folder.",
                )
                return

        if not self.get_parameter_value("generate_captions") and (images and len(images) > 0):
            captions = self.get_parameter_value("captions")
            if not captions or len(captions) < len(images):
                self._set_status_results(
                    was_successful=False,
                    result_details="Number of images and captions do not match. Please provide more captions or enable caption generation.",
                )
                return

        logger.debug(f"Processing {len(images)} images")

        try:
            self.create_dataset(dataset_folder_path, images)
            logger.debug(f"Dataset created at: {dataset_folder_path}")
        except Exception as e:
            logger.exception(f"Error while creating dataset: {str(e)}")
            self._set_status_results(was_successful=False, result_details=f"Error while creating dataset: {str(e)}")
            return

        try:
            dataset_toml_path = self.generate_toml(
                output_folder=dataset_folder_path, resolution=image_resolution, num_repeats=num_repeats
            )
            logger.debug(f"Dataset configuration file generated at: {dataset_toml_path}")
        except Exception as e:
            logger.exception(f"Error while generating dataset.toml: {str(e)}")
            self._set_status_results(
                was_successful=False, result_details=f"Error while generating dataset.toml: {str(e)}"
            )
            return

        self.set_parameter_value("dataset_config_path", str(dataset_toml_path))
        self._set_status_results(was_successful=True, result_details="Success")
