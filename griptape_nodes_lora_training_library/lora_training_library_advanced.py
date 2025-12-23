import logging
from pathlib import Path

import pygit2

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lora_training_library")


class LoraTrainingLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for LoRA Training."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library."""
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        logger.info("Initializing sd-scripts submodule...")
        self._init_sd_scripts_submodule()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)

    def _get_library_root(self) -> Path:
        """Get the library root directory (where .venv lives)."""
        return Path(__file__).parent

    def _update_submodules_recursive(self, repo_path: Path) -> None:
        """Recursively update and initialize all submodules.

        Equivalent to: git submodule update --init --recursive
        """
        repo = pygit2.Repository(str(repo_path))
        repo.submodules.update(init=True)

        # Recursively update nested submodules
        for submodule in repo.submodules:
            submodule_path = repo_path / submodule.path
            if submodule_path.exists() and (submodule_path / ".git").exists():
                self._update_submodules_recursive(submodule_path)

    def _init_sd_scripts_submodule(self) -> Path:
        """Initialize the sd-scripts git submodule."""
        library_root = self._get_library_root()
        sd_scripts_submodule_dir = library_root / "sd-scripts"

        if sd_scripts_submodule_dir.exists() and any(sd_scripts_submodule_dir.iterdir()):
            logger.info("sd-scripts submodule already initialized")
            return sd_scripts_submodule_dir

        git_repo_root = library_root.parent
        self._update_submodules_recursive(git_repo_root)

        if not sd_scripts_submodule_dir.exists() or not any(sd_scripts_submodule_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {sd_scripts_submodule_dir} is empty or does not exist"
            )

        logger.info("sd-scripts submodule initialized successfully")
        return sd_scripts_submodule_dir
