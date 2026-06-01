from pathlib import Path
from typing import Final

DOCKER_IMAGE_PREFIX: Final[str] = "docker.io/jefzda/sweap-images"
DEFAULT_DOCKERHUB_USERNAME: Final[str] = "jefzda"
SOURCE_REPO_PATH: Final[str] = "/app"
HARNESS_SUBMODULE_PATH: Final[Path] = Path(__file__).parent / "SWE-bench_Pro-os"
OFFICIAL_HARNESS_REPO: Final[str] = "https://github.com/scaleapi/SWE-bench_Pro-os"
OFFICIAL_HARNESS_REF: Final[str] = "0c64e26f00b9c190432de7fc520c8ceed5c25518"
