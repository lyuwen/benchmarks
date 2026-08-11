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
    "find . -path ./x",             # -path is fine
    "git branch",                   # bare list
    "git branch --list",            # explicit list
    "git branch -a",                # list all
    "git branch -r",                # list remotes
    "git branch -v",                # verbose list
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


@pytest.mark.parametrize("cmd", [
    # 1. find exec/delete vectors
    "find . -delete",
    "find . -exec rm {} +",
    "find . -exec rm {} \\;",
    "find . -execdir rm {} +",
    "find . -ok rm {} \\;",
    "find . -okdir rm {} \\;",
    "find . -fprintf out.txt '%p'",
    "find . -fprint out.txt",
    "find . -fprint0 out.txt",
    "find . -fls out.txt",
    # 2. python/python3 are exec vectors
    "python x.py",
    "python script.py",
    "python -m http.server",
    "python -m mod",
    "python3 -c'code'",
    'python -c "import os"',
    # 3. command substitution / backticks / process substitution
    "cat $(rm -rf /)",
    "cat `rm -rf /`",
    "echo ${HOME}",
    "cat <(rm -rf /)",
    "cat >(rm -rf /)",
    # 4. env / awk exec vectors
    "env rm -rf /",
    'awk \'BEGIN{system("rm -rf /")}\'',
    # 5. git branch write flags / branch creation
    "git branch -D x",
    "git branch -d x",
    "git branch -m a b",
    "git branch -M a b",
    "git branch newname",
    "git branch --delete x",
    "git branch --set-upstream-to=origin/main",
    "git branch --force x y",
    # 6. newline as segment separator
    "cat a\nrm b",
])
def test_rejects_bypasses(cmd):
    assert is_readonly_command(cmd) is False
