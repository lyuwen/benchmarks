import os

OUTPUT_FILENAME = os.getenv("OUTPUT_FILENAME", "output.jsonl")
EVAL_AGENT_SERVER_IMAGE = os.getenv("EVAL_AGENT_SERVER_IMAGE", "ghcr.nju.edu.cn/openhands/eval-agent-server")
