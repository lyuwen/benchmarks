import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


def _get_submodule_sha(submodule_path: Path) -> str:
    result = subprocess.run(
        ["git", "submodule", "status", str(submodule_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    sha = result.stdout.strip().split()[0].lstrip("+-")
    return sha


def get_sdk_sha() -> str:
    """
    Get the current git sha from the SDK submodule.
    """
    return _get_submodule_sha(PROJECT_ROOT / "vendor" / "software-agent-sdk")


SDK_SHA = get_sdk_sha()
SDK_SHA = "f011127916917cb3a3a2c29d2c5b151ef2464f44" # Fix SDK version to freeze agent server images
SDK_SHORT_SHA = SDK_SHA[:7]
