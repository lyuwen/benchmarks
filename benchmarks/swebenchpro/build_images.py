#!/usr/bin/env python3
"""Build agent-server images for SWE-Bench Pro instances."""

import hashlib
import re
import sys

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


def get_official_docker_image(dockerhub_tag: str) -> str:
    tag = dockerhub_tag.strip()
    if not tag:
        raise ValueError("dockerhub_tag must not be empty")
    return f"{constants.DOCKER_IMAGE_PREFIX}:{tag}"


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
) -> list[str]:
    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    return sorted(
        {get_official_docker_image(str(row["dockerhub_tag"])) for _, row in df.iterrows()}
    )


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    parser.set_defaults(dataset="ScaleAI/SWE-bench_Pro", split="test")
    args = parser.parse_args(argv)

    base_images = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
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
