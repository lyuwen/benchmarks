# benchmarks/scaleswe_interactive/tests/test_readonly_guard.py
import pytest
from benchmarks.scaleswe_interactive.readonly_guard import is_readonly_command


@pytest.mark.parametrize("cmd", [
    "cat foo.py",
    "grep -rn needle src/",
    "ls -la",
    "git --no-pager log --oneline -10",
    "git --no-pager diff HEAD~1",
    "find . -name '*.py'",
    "head -50 a.py",
    "sed -n '1,80p' a.py",          # read-only sed (no -i)
    "cat a.py | grep x | head -5",  # all segments read-only
])
def test_allows_readonly(cmd):
    assert is_readonly_command(cmd) is True


@pytest.mark.parametrize("cmd", [
    "rm -rf /",
    "echo hi > f",
    "cat a >> b",
    "sed -i 's/a/b/' f",
    "git commit -m x",
    "git apply p.patch",
    "python -c 'open(1)'",
    "tee f",
    "mv a b",
    "cp a b",
    "cat a && rm b",       # chained: one writer taints all
    "grep x f; rm y",      # semicolon chain with writer
    "truncate -s0 f",
])
def test_rejects_writers(cmd):
    assert is_readonly_command(cmd) is False
