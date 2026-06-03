#!/usr/bin/env python3
"""Build agent-server images for SWE-Bench Pro instances."""

import hashlib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.swebenchpro import constants
from benchmarks.utils.build_utils import (
    build_all_images,
    default_build_output_dir,
    get_build_parser,
)
from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger

logger = get_logger(__name__)
MAX_CUSTOM_TAG_LENGTH = 96
CUSTOM_TAG_SANITIZER = re.compile(r"[^a-z0-9_.-]+")


def get_official_docker_image(dockerhub_tag: str, image_prefix: str | None = None) -> str:
    tag = dockerhub_tag.strip()
    if not tag:
        raise ValueError("dockerhub_tag must not be empty")
    prefix = image_prefix or constants.DOCKER_IMAGE_PREFIX
    return f"{prefix}:{tag}"


def extract_custom_tag(base_image: str) -> str:
    before, sep, tag = base_image.rpartition(":")
    if not sep or not tag:
        raise ValueError(f"Could not extract docker tag from image: {base_image}")

    sanitized = CUSTOM_TAG_SANITIZER.sub("-", tag.lower()).strip(".-")
    if not sanitized:
        sanitized = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]
    if len(sanitized) <= MAX_CUSTOM_TAG_LENGTH:
        return sanitized

    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:12]
    prefix = sanitized[: MAX_CUSTOM_TAG_LENGTH - len(digest) - 1].rstrip(".-")
    return f"{prefix}-{digest}"


def collect_unique_base_images(
    dataset: str,
    split: str,
    n_limit: int,
    selected_instances_file: str | None = None,
    image_prefix: str | None = None,
) -> list[str]:
    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    return sorted(
        {get_official_docker_image(str(row["dockerhub_tag"]), image_prefix) for _, row in df.iterrows()}
    )


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    parser.set_defaults(dataset="ScaleAI/SWE-bench_Pro", split="test")
    parser.add_argument(
        "--docker-image-prefix",
        type=str,
        default=None,
        help=(
            f"Override Docker image repository prefix (default: {constants.DOCKER_IMAGE_PREFIX}). "
            "Replaces everything before the last ':' in the image reference. "
            "E.g., 'myregistry.com/myorg/sweap-images' turns 'docker.io/jefzda/sweap-images:tag' "
            "into 'myregistry.com/myorg/sweap-images:tag'."
        ),
    )
    args = parser.parse_args(argv)

    base_images = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
        args.docker_image_prefix,
    )
    build_dir = default_build_output_dir(args.dataset, args.split)

    return build_all_images(
        base_images=base_images,
        target=args.target,
        build_dir=build_dir,
        image=args.image,
        push=args.push,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
        base_image_to_custom_tag_fn=extract_custom_tag,
        post_build_fn=None,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
