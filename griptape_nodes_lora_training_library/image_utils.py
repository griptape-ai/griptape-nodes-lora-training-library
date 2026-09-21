from urllib.error import URLError

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.loaders import ImageLoader
from griptape_nodes.files.file import File, FileLoadError
from requests.exceptions import RequestException


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
