#!/usr/bin/env python3
"""
Build agent-server images for all unique R2E-Gym base images in a dataset split.
Example:
  uv run benchmarks/r2e-gym/build_images.py \
    --dataset /mnt/huawei/users/lfu/datasets/R2E-Gym-Subset --split train \
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

def get_base_image(row) -> str:
    """
    Determine the base image from a dataset row.
    Adapting for R2E-Gym: expecting 'base_image' column or deriving from 'instance_id'.
    """
    # Option A: If the dataset has a 'base_image' column, use it.
    if "base_image" in row:
        return row["base_image"]
    
    # Option B: R2E-Gym-Subset appears to use 'docker_image'
    if "docker_image" in row:
        return row["docker_image"]

    # Option C: Default behavior similar to SWE-Gym - construct from instance_id
    # You might need to adjust this prefix based on where R2E attributes are stored.
    # For now, we assume a standard naming convention or placeholder.
    instance_id = str(row["instance_id"])
    # Example placeholder logic:
    # return f"docker.io/r2e-gym/env:{instance_id}"
    
    # Returning a clear error if we can't determine it, to prompt user intervention
    logger.warning(f"Could not determine base image for instance {instance_id}. Using placeholder logic.")
    return f"docker.io/r2e-gym/{instance_id}:latest"


def extract_custom_tag(base_image: str) -> str:
    """
    Extract a custom tag from the base image name, including the image tag
    for uniqueness (R2E-Gym images share repo names but differ by commit SHA tag).
    """
    parts = base_image.split("/")[-1]  # e.g. "scrapy_final:0c5087..."
    name, _, tag = parts.partition(":")
    if tag:
        return f"{name}-{tag[:12]}"  # e.g. "scrapy_final-0c50879568de"
    return name


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
    
    base_images = set()
    for _, row in df.iterrows():
        base_image = get_base_image(row)
        base_images.add(base_image)
        
    return sorted(base_images)


def main(argv: list[str]) -> int:
    parser = get_build_parser()
    args = parser.parse_args(argv)

    base_images = collect_unique_base_images(
        args.dataset,
        args.split,
        args.n_limit,
        args.select,
    )
    
    logger.info(f"Collected {len(base_images)} unique base images.")
    
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
