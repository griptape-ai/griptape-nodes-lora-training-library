from pathlib import Path
from urllib.error import URLError

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.loaders import ImageLoader
from griptape_nodes.files.file import File, FileLoadError
from requests.exceptions import RequestException

# Characters that are illegal in filenames on Windows (and confusing elsewhere).
_UNSAFE_FILENAME_CHARS = frozenset('<>:"/\\|?*')


def load_image_from_url_artifact(image_url_artifact: ImageUrlArtifact) -> ImageArtifact:
    """Load an ImageArtifact from an ImageUrlArtifact with proper error handling.

    Reads through the engine's file layer so that macro paths (e.g.
    `{outputs}/images/foo.png`) and plain filesystem paths resolve, in addition
    to `http(s)://` URLs.

    Args:
        image_url_artifact: The ImageUrlArtifact to load

    Returns:
        ImageArtifact: The loaded image artifact

    Raises:
        ValueError: If image download fails with descriptive error message
    """
    try:
        image_bytes = File(image_url_artifact.value).read_bytes()
    except (URLError, RequestException, ConnectionError, TimeoutError, OSError, FileLoadError) as err:
        details = (
            f"Failed to download image at '{image_url_artifact.value}'.\n"
            f"If this workflow was shared from another engine installation, "
            f"that image file will need to be regenerated.\n"
            f"Error: {err}"
        )
        raise ValueError(details) from err

    return ImageLoader().parse(image_bytes)


def load_image_from_path(image_path: str | Path) -> ImageArtifact:
    """Load an ImageArtifact from a file path.

    The read goes through the engine's file layer, so plain absolute/relative paths,
    engine macro paths (e.g. `{outputs}/images/foo.png`) and URLs are all accepted.
    The resulting artifact keeps the file's own name, so a dataset built from paths
    ends up with the original filenames rather than generated ones.

    Args:
        image_path: Path to the image file

    Returns:
        ImageArtifact: The loaded image artifact

    Raises:
        ValueError: If the image cannot be read with a descriptive error message
    """
    file = File(str(image_path))
    try:
        image_bytes = file.read_bytes()
    except (FileLoadError, URLError, RequestException, ConnectionError, TimeoutError, OSError) as err:
        details = f"Failed to load image at '{image_path}'.\nError: {err}"
        raise ValueError(details) from err

    image_artifact = ImageLoader().parse(image_bytes)
    filename = file.name
    # Keep the original filename only when it is usable as-is. A path that came in as a
    # URL can leave query/fragment characters in the name that some filesystems reject;
    # in that case fall back to the artifact's own generated name.
    if filename and not set(filename) & _UNSAFE_FILENAME_CHARS:
        image_artifact.name = filename
    return image_artifact
