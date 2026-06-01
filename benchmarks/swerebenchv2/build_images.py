#!/usr/bin/env python3
"""
Build agent-server images for all unique SWE-rebench-V2 base images in a dataset split.

Example:
  uv run benchmarks/swerebenchv2/build_images.py \
    --dataset thirdparty/SWE-rebench-V2/SWE-rebench-V2-py.jsonl --split train \
    --image ghcr.io/openhands/eval-agent-server --target source-minimal
"""

import sys

from benchmarks.utils.build_utils import (
    build_all_images,
    default_build_output_dir,
    get_build_parser,
)
from benchmarks.utils.dataset import get_dataset
from openhands.sdk import get_logger


logger = get_logger(__name__)


def get_official_docker_image(image_name: str) -> str:
    """Normalize SWE-rebench-V2 image name to a full Docker reference.

    Images in the dataset are stored as e.g.:
        docker.io/swerebenchv2/unidata-netcdf-c:1925-ad6bff3
    If the prefix is missing, prepend docker.io/.
    """
    official_image_name: str = image_name.lower().strip()
    if not official_image_name.startswith("docker.io"):
        official_image_name = f"docker.io/{official_image_name}"
    logger.debug(f"Official SWE-rebench-V2 image: {official_image_name}")
    return official_image_name


def extract_custom_tag(base_image: str) -> str:
    """Extract a unique identifier from image name for use in agent-server tags.

    Replaces ':' with '-' since Docker tags cannot contain colons.

    Example:
        docker.io/swerebenchv2/unidata-netcdf-c:1925-ad6bff3
        -> unidata-netcdf-c-1925-ad6bff3
    """
    name_tag = base_image.split("/")[-1]
    return name_tag.replace(":", "-")


def collect_unique_base_images(
    dataset,
    split,
    n_limit,
    selected_instances_file: str | None = None,
):
    df = get_dataset(
        dataset_name=dataset,
        split=split,
        eval_limit=n_limit if n_limit else None,
        selected_instances_file=selected_instances_file,
    )
    df = df[df["problem_statement"].str.strip().astype(bool)]
    return sorted(
        {get_official_docker_image(str(row["image_name"])) for _, row in df.iterrows()}
    )


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    args = parser.parse_args(argv)

    base_images: list[str] = collect_unique_base_images(
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
