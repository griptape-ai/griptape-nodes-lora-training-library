from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path
from huggingface_hub import scan_cache_dir

if TYPE_CHECKING:
    from train_lora.train_lora_node import TrainLoraNode
    from griptape_nodes.common.parameters.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

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
    def get_script_params(self) -> list[str]:
        raise NotImplementedError
    
    @abstractmethod
    def get_mixed_precision(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def get_script_name(self) -> str:
        raise NotImplementedError
    
    def _get_repo_snapshot_path(self, repo_id: str, revision: str) -> Path:
        """Get the path to a specific repo revision's snapshot directory in HuggingFace cache."""
        cache_info = scan_cache_dir()

        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                for rev in repo.revisions:
                    if rev.commit_hash == revision:
                        return Path(rev.snapshot_path)

        raise FileNotFoundError(
            f"Repository '{repo_id}' with revision '{revision}' not found in HuggingFace cache. "
            f"Please download the model using 'huggingface-cli download {repo_id}'."
        )

    def _find_file_in_snapshot(self, snapshot_path: Path, patterns: list[str]) -> Path:
        """
        Search for files matching any of the given patterns in the snapshot directory.

        Patterns can be:
        - Exact filenames: "ae.safetensors"
        - Glob patterns: "flux*.safetensors"
        - Paths with globs: "vae/*.safetensors"
        - Sharded models: "text_encoder_2/model-00001-of-*.safetensors" (will be merged)

        Returns the first match, preferring files with shorter names.
        For sharded model patterns, automatically merges shards into a single file.
        """
        for pattern in patterns:
            # Check if this is a shard pattern (contains "model-00001-of-")
            if "model-00001-of-" in pattern:
                # Get the directory to search in
                if '/' in pattern:
                    search_dir = snapshot_path / pattern.rsplit('/', 1)[0]
                else:
                    search_dir = snapshot_path

                if search_dir.exists():
                    # Extract the pattern after the directory
                    shard_pattern = pattern.split('/')[-1] if '/' in pattern else pattern
                    shard_files = list(search_dir.glob(shard_pattern))
                    if shard_files:
                        logger.info(f"Found sharded model matching {pattern}")
                        return self._merge_sharded_safetensors(shard_files[0])
                continue

            # Try exact match first
            file_path = snapshot_path / pattern
            if file_path.exists() and file_path.is_file():
                return file_path

            # Try glob pattern
            matches = [m for m in snapshot_path.glob(pattern) if m.is_file()]
            if matches:
                # Return the first match (prefer shorter names for main models)
                matched_path = sorted(matches, key=lambda p: len(p.name))[0]
                return matched_path

            # Try recursive glob if pattern doesn't already include **
            if '**' not in pattern:
                matches = [m for m in snapshot_path.rglob(pattern) if m.is_file()]
                if matches:
                    matched_path = sorted(matches, key=lambda p: len(p.name))[0]
                    return matched_path

        raise FileNotFoundError(
            f"No files matching patterns {patterns} found in snapshot {snapshot_path}"
        )

    def _merge_sharded_safetensors(self, first_shard_path: Path) -> Path:
        """
        Merge sharded safetensors files into a single file.

        Args:
            first_shard_path: Path to the first shard file (e.g., model-00001-of-00002.safetensors)

        Returns:
            Path to the merged model.safetensors file
        """
        from safetensors.torch import load_file, save_file
        import re
        import gc

        output_safetensors = first_shard_path.parent / "model.safetensors"

        # If merged file already exists, return it
        if output_safetensors.exists():
            logger.info(f"Using existing merged safetensors file: {output_safetensors}")
            return output_safetensors

        # Parse the shard pattern (e.g., model-00001-of-00002.safetensors)
        match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", first_shard_path.name)
        if not match:
            raise ValueError(f"Unexpected shard filename format: {first_shard_path.name}")

        prefix = match.group(1)
        total_shards = int(match.group(3))

        logger.info(f"Merging {total_shards} safetensors shards into {output_safetensors.name}...")
        logger.info(f"This is a one-time operation and may take a minute...")

        # Find all shard files
        shard_files = []
        for i in range(1, total_shards + 1):
            shard_file = first_shard_path.parent / f"{prefix}{i:05d}-of-{total_shards:05d}.safetensors"
            if not shard_file.exists():
                raise FileNotFoundError(f"Missing shard file: {shard_file}")
            shard_files.append(shard_file)
            logger.info(f"  Found shard {i}/{total_shards}: {shard_file.name}")

        # Merge all shards into one state dict
        merged_state_dict = {}
        for shard_file in shard_files:
            logger.info(f"Loading {shard_file.name}...")
            shard_data = load_file(str(shard_file))
            merged_state_dict.update(shard_data)
            del shard_data
            gc.collect()

        logger.info(f"Saving merged model to {output_safetensors}...")
        save_file(merged_state_dict, str(output_safetensors))

        # Clean up
        del merged_state_dict
        gc.collect()

        logger.info(f"✓ Successfully merged {total_shards} shards into: {output_safetensors}")
        return output_safetensors

    def get_model_file_path(self, patterns: list[str], repo_parameter: HuggingFaceRepoParameter) -> Path:
        """
        Get the path to a model file in the HuggingFace cache.

        Args:
            patterns: List of file patterns to search for
            repo_parameter: HuggingFaceRepoParameter to use

        Returns:
            Path to the model file
        """

        repo_id, revision = repo_parameter.get_repo_revision()
        snapshot_path = self._get_repo_snapshot_path(repo_id, revision)
        model_path = self._find_file_in_snapshot(snapshot_path, patterns)
        if not model_path:
            raise FileNotFoundError(
                f"Could not find model file in {snapshot_path}. "
                f"Expected a file matching patterns: {', '.join(patterns)}"
            )
        return model_path
