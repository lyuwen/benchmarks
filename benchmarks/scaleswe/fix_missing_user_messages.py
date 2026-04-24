"""Fix JSONL history files that are missing the initial user message.

Due to a race condition in RemoteConversation (WebSocket subscription not
ready before the first send_message), the user instruction can be dropped
from the local event cache and thus from the dumped history.

This script reconstructs the instruction from the source dataset and the
prompt template, then inserts it after the initial system message.

Usage:
    python -m benchmarks.scaleswe.fix_missing_user_messages \
        --input history.jsonl \
        --output history_fixed.jsonl \
        --dataset thirdparty/Scale-SWE/scale-swe-batch1.jsonl \
        --prompt-path benchmarks/scaleswe/prompts/default.j2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def load_source_dataset(dataset_path: str) -> dict[str, dict]:
    """Load a JSONL dataset file and index by instance_id."""
    instances: dict[str, dict] = {}
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instances[record["instance_id"]] = record
    return instances


def render_instruction(instance: dict, template_path: str) -> str:
    """Render the prompt template with instance data.

    Mirrors the logic in benchmarks.scaleswe.run_infer.get_instruction
    without requiring EvalMetadata.
    """
    prompts_dir = str(Path(template_path).parent)
    template_name = Path(template_path).name
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    repo_path = instance.get("workdir", "/workspace")
    if not repo_path.endswith("/"):
        repo_path += "/"

    inst = dict(instance)
    inst["repo_path"] = repo_path
    inst.setdefault("base_commit", inst.get("parent_commit", ""))
    if "repo" not in inst or "/" not in inst.get("repo", ""):
        user = inst.get("user", "")
        repo = inst.get("repo", "")
        if user and repo:
            inst["repo"] = f"{user}/{repo}"

    context = {
        "instance": inst,
        "workspace_dir_name": inst.get("repo", "").split("/")[-1],
        "actual_workspace_path": "/workspace",
        "metadata": None,
        "test_instructions": "",
    }
    return template.render(context)


def has_user_message_after_system(messages: list[dict]) -> bool:
    """Check if there is a user message right after the initial system message."""
    if not messages:
        return False
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                return True
            return False
    return False


def fix_row(
    row: dict,
    source_dataset: dict[str, dict],
    template_path: str,
) -> tuple[dict, bool]:
    """Fix a single JSONL row if user message is missing.

    Returns (fixed_row, was_modified).
    """
    instance_id = row.get("instance_id", "")
    messages = row.get("messages", [])

    if has_user_message_after_system(messages):
        return row, False

    if not instance_id:
        print("WARNING: row missing instance_id, skipping", file=sys.stderr)
        return row, False

    instance_data = source_dataset.get(instance_id)
    if instance_data is None:
        print(f"WARNING: instance {instance_id} not found in source dataset, skipping", file=sys.stderr)
        return row, False

    instruction = render_instruction(instance_data, template_path)
    user_msg = {"role": "user", "content": instruction}

    fixed_messages = list(messages)
    insert_idx = 0
    for i, msg in enumerate(fixed_messages):
        if msg.get("role") == "system":
            insert_idx = i + 1
            break

    fixed_messages.insert(insert_idx, user_msg)
    row = dict(row)
    row["messages"] = fixed_messages
    return row, True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix JSONL history files missing user messages"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL file to fix"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL file path"
    )
    parser.add_argument(
        "--dataset",
        default="thirdparty/Scale-SWE/scale-swe-batch1.jsonl",
        help="Source dataset JSONL for instance data lookup",
    )
    parser.add_argument(
        "--prompt-path",
        default=str(
            (Path(__file__).parent / "prompts" / "default.j2").resolve()
        ),
        help="Path to prompt template file",
    )
    args = parser.parse_args()

    source_dataset = load_source_dataset(args.dataset)
    print(f"Loaded {len(source_dataset)} instances from {args.dataset}")

    total = 0
    fixed = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: invalid JSON on line {line_num}: {e}", file=sys.stderr)
                continue

            row, was_modified = fix_row(row, source_dataset, args.prompt_path)
            total += 1
            if was_modified:
                fixed += 1

            fout.write(json.dumps(row) + "\n")

    print(f"Processed {total} rows, fixed {fixed} missing user messages")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
