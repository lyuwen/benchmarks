"""Package manager mirror configuration for faster installations.

This module provides environment variable configurations for various package managers
to use mirror repositories, significantly speeding up installations in certain regions.
"""

from __future__ import annotations

import os
from typing import Dict, List

# Predefined mirror configurations
MIRROR_CONFIGS: Dict[str, List[str]] = {
    "china": [
        # Python package mirrors
        "export PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple",
        "export PIP_TRUSTED_HOST=mirrors.ustc.edu.cn",
        "export UV_INDEX=https://mirrors.ustc.edu.cn/pypi/simple",
        # Node.js package mirrors
        "export NPM_CONFIG_REGISTRY=https://registry.npmmirror.com",
        # Go module mirrors
        "export GO111MODULE=on",
        "export GOPROXY=https://goproxy.cn",
    ],
    "tsinghua": [
        # Python package mirrors (Tsinghua University)
        "export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
        "export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn",
        "export UV_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple",
        # Node.js package mirrors
        "export NPM_CONFIG_REGISTRY=https://registry.npmmirror.com",
        # Go module mirrors
        "export GO111MODULE=on",
        "export GOPROXY=https://goproxy.cn",
    ],
    "aliyun": [
        # Python package mirrors (Alibaba Cloud)
        "export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple",
        "export PIP_TRUSTED_HOST=mirrors.aliyun.com",
        "export UV_INDEX=https://mirrors.aliyun.com/pypi/simple",
        # Node.js package mirrors
        "export NPM_CONFIG_REGISTRY=https://registry.npmmirror.com",
        # Go module mirrors
        "export GO111MODULE=on",
        "export GOPROXY=https://goproxy.cn,direct",
    ],
}


def get_mirror_env_commands(mirror: str | None = None) -> List[str]:
    """Get environment setup commands for package manager mirrors.

    Args:
        mirror: Mirror configuration name. If None, checks PACKAGE_MIRROR env var.
                Valid values: "china", "tsinghua", "aliyun", or None for default.

    Returns:
        List of export commands to set up package manager mirrors.

    Examples:
        >>> # Via environment variable
        >>> os.environ["PACKAGE_MIRROR"] = "china"
        >>> commands = get_mirror_env_commands()

        >>> # Via argument
        >>> commands = get_mirror_env_commands("china")
    """
    if mirror is None:
        mirror = os.getenv("PACKAGE_MIRROR", "").strip().lower()

    if not mirror or mirror == "default":
        return []

    if mirror not in MIRROR_CONFIGS:
        available = ", ".join(MIRROR_CONFIGS.keys())
        raise ValueError(
            f"Unknown mirror configuration: {mirror}. "
            f"Available options: {available}"
        )

    return MIRROR_CONFIGS[mirror].copy()


def format_mirror_env_exports(mirror: str | None = None) -> str:
    """Format mirror environment variables as a shell script block.

    Args:
        mirror: Mirror configuration name or None.

    Returns:
        Shell script snippet with export commands, or empty string if no mirror.

    Example:
        >>> script = format_mirror_env_exports("china")
        >>> print(script)
        export PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
        export PIP_TRUSTED_HOST=mirrors.ustc.edu.cn
        ...
    """
    commands = get_mirror_env_commands(mirror)
    if not commands:
        return ""
    return "\n".join(commands) + "\n"
