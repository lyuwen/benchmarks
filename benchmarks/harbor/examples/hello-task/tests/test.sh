#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
content="$(cat /answer.txt 2>/dev/null || true)"
if [ "$content" = "hello harbor" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo "unexpected content: [$content]" >&2
  echo 0 > /logs/verifier/reward.txt
fi
