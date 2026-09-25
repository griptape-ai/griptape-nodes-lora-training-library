import logging
import subprocess
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lora_training_library")


class LoraTrainingLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for LoRA Training."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library."""
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        # The sd-scripts submodule populates the execution environment (it's imported
        # only by process()), so only the worker needs to initialize it.
        if not GriptapeNodes.LibraryManager().is_worker:
            return

        logger.info("Initializing sd-scripts submodule...")
        self._init_sd_scripts_submodule()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)

    def _get_library_root(self) -> Path:
        """Get the library root directory (where .venv lives)."""
        return Path(__file__).parent

    def _init_sd_scripts_submodule(self) -> Path:
        """Initialize the sd-scripts git submodule."""
        library_root = self._get_library_root()
        sd_scripts_submodule_dir = library_root / "sd-scripts"

        if sd_scripts_submodule_dir.exists() and any(sd_scripts_submodule_dir.iterdir()):
            logger.info("sd-scripts submodule already initialized")
            return sd_scripts_submodule_dir

        # The git CLI rather than pygit2: the engine dropped pygit2 (its bundled TLS trust
        # store breaks on some platforms) and requires git on PATH, so it is the one tool
        # guaranteed to be here.
        git_repo_root = library_root.parent
        subprocess.check_call(["git", "-C", str(git_repo_root), "submodule", "update", "--init", "--recursive"])

        if not sd_scripts_submodule_dir.exists() or not any(sd_scripts_submodule_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {sd_scripts_submodule_dir} is empty or does not exist"
            )

        logger.info("sd-scripts submodule initialized successfully")
        return sd_scripts_submodule_dir
