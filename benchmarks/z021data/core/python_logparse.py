"""Python test-log parsing for gen_test validation.

Extracted from f2p_validation.py so the log-parsing layer can evolve
independently (e.g. add unittest/nose/trial parsers later). Everything here is
pure text-in / dict-out; there is no docker or subprocess dependency.

Public API:
  parse_pytest_output(output) -> dict
  parse_run_output(runner, output) -> dict
  compute_transitions(runner, before, after) -> dict

A parse result dict has the shape:
  {
    'passed':  [node_id, ...],
    'failed':  [node_id, ...],
    'errors':  [node_id, ...],
    'passed_count': int,
    'failed_count': int,
    'error_count':  int,
  }
"""

from __future__ import annotations

import re

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m|\033\[[0-9;]*m')


def _empty_result() -> dict:
    return {'passed': [], 'failed': [], 'errors': [],
            'passed_count': 0, 'failed_count': 0, 'error_count': 0}


def parse_pytest_output(output: str) -> dict:
    """Parse pytest console output into per-test node ids + summary counts.

    Recognises both orderings of the verbose per-test line:
        tests/test_x.py::test_y PASSED
        PASSED tests/test_x.py::test_y
    If no per-test lines are present (e.g. quiet mode) it falls back to the
    summary line (``3 passed, 1 failed in 0.2s``) for counts only.
    """
    passed, failed, errors = [], [], []
    clean = _ANSI_ESCAPE.sub('', output or "")

    for m in re.finditer(
        r'^([\w/\.\-]+::\S+(?:\[.*?\])?)\s+(PASSED|FAILED|ERROR)', clean, re.MULTILINE
    ):
        name, status = m.group(1), m.group(2)
        if status == 'PASSED':
            passed.append(name)
        elif status == 'FAILED':
            failed.append(name)
        elif status == 'ERROR':
            errors.append(name)

    for m in re.finditer(
        r'^(PASSED|FAILED|ERROR)\s+([\w/\.\-]+::\S+(?:\[.*?\])?)', clean, re.MULTILINE
    ):
        status, name = m.group(1), m.group(2)
        if status == 'PASSED' and name not in passed:
            passed.append(name)
        elif status == 'FAILED' and name not in failed:
            failed.append(name)
        elif status == 'ERROR' and name not in errors:
            errors.append(name)

    # A genuine pytest terminal summary always carries the elapsed-time suffix
    # (``... in 0.62s``). Requiring it prevents arbitrary log text such as
    # ``port 5432 failed`` (DB connection errors) from being misread as a
    # summary line, which previously produced counts like ``failed=5432``.
    summary_line = ""
    for line in clean.splitlines():
        if re.search(
            r'\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)\b'
            r'.*\bin\s+\d+(?:\.\d+)?\s*s(?:econds?)?\b',
            line,
        ):
            summary_line = line

    # The terminal summary line is the AUTHORITATIVE count source. In pytest's
    # default / short reporting mode the per-test lines only list failures and
    # errors (passed tests are NOT printed), so relying on the per-test lists
    # would undercount ``passed`` (e.g. summary says ``11 passed`` but no PASSED
    # line exists). We therefore always parse the summary when present and take
    # the max of (summary count, per-test list length) per category. The node-id
    # lists are still returned for downstream F2P node matching.
    summary_counts = None
    if summary_line:
        label_map = {
            'passed': 'passed', 'pass': 'passed',
            'failed': 'failed', 'fail': 'failed', 'failure': 'failed', 'failures': 'failed',
            'error': 'errors', 'errors': 'errors',
        }
        counts = {'passed': 0, 'failed': 0, 'errors': 0}
        for m in re.finditer(r'(\d+)\s+(\w+)', summary_line):
            fn = label_map.get(m.group(2).lower())
            if fn:
                counts[fn] = int(m.group(1))
        if any(counts.values()):
            summary_counts = counts

    if summary_counts is not None:
        return {'passed': passed, 'failed': failed, 'errors': errors,
                'passed_count': max(summary_counts['passed'], len(passed)),
                'failed_count': max(summary_counts['failed'], len(failed)),
                'error_count': max(summary_counts['errors'], len(errors))}

    return {'passed': passed, 'failed': failed, 'errors': errors,
            'passed_count': len(passed), 'failed_count': len(failed),
            'error_count': len(errors)}


def parse_unittest_output(output: str) -> dict:
    """Parse unittest console output into per-test node ids + summary counts.

    Recognises unittest verbose output format:
        test_name (module.ClassName.test_name) ... ok
        test_name (module.ClassName.test_name) ... FAIL
        test_name (module.ClassName.test_name) ... ERROR

    Summary line formats:
        OK
        FAILED (failures=1)
        FAILED (errors=1)
        FAILED (failures=1, errors=1)
        Ran N tests in X.XXXs
    """
    passed, failed, errors = [], [], []
    clean = _ANSI_ESCAPE.sub('', output or "")

    # Parse per-test lines: test_name (full.qualified.name) ... STATUS
    # The full qualified name in parens is the node_id we want
    for m in re.finditer(
        r'^\s*\S+\s+\(([^)]+)\)\s+\.\.\.\s+(ok|FAIL|ERROR)', clean, re.MULTILINE
    ):
        node_id, status = m.group(1), m.group(2)
        # Convert module.Class.test_name to module.py::Class::test_name format
        # to match pytest-style node ids for consistency
        parts = node_id.split('.')
        if len(parts) >= 2:
            # Assume last part is test method, second-to-last is class, rest is module
            test_method = parts[-1]
            class_name = parts[-2] if len(parts) >= 2 else ''
            module_parts = parts[:-2] if len(parts) > 2 else parts[:-1]

            # Build pytest-style node_id
            if module_parts:
                module_path = '/'.join(module_parts) + '.py'
            else:
                module_path = 'tests.py'

            if class_name and not class_name.startswith('test_'):
                pytest_node_id = f"{module_path}::{class_name}::{test_method}"
            else:
                pytest_node_id = f"{module_path}::{test_method}"
        else:
            # Fallback: use the node_id as-is
            pytest_node_id = node_id

        if status == 'ok':
            passed.append(pytest_node_id)
        elif status == 'FAIL':
            failed.append(pytest_node_id)
        elif status == 'ERROR':
            errors.append(pytest_node_id)

    # ---- Summary line parsing (authoritative totals) ----
    # unittest always prints "Ran N tests in X.XXs" then a status line:
    #   OK
    #   OK (skipped=2, expected failures=1)
    #   FAILED (failures=1, errors=2, skipped=3)
    ran_count = 0
    for line in clean.splitlines():
        m = re.search(r'Ran\s+(\d+)\s+tests?\s+in', line)
        if m:
            ran_count = int(m.group(1))
            break

    sm_failed = sm_errors = sm_skipped = sm_xfail = 0
    for line in clean.splitlines():
        m = re.match(r'\s*(?:OK|FAILED)\b(?:\s*\(([^)]*)\))?\s*$', line)
        if not m:
            continue
        body = m.group(1) or ''
        for key, val in re.findall(r'([A-Za-z][A-Za-z ]*?)\s*=\s*(\d+)', body):
            key, n = key.strip().lower(), int(val)
            # Order matters: 'expected failure' contains 'failure'.
            if 'expected failure' in key:
                sm_xfail = n
            elif 'unexpected success' in key:
                sm_failed += n            # unittest treats these as failures
            elif 'failure' in key:
                sm_failed += n
            elif 'error' in key:
                sm_errors = n
            elif 'skip' in key:
                sm_skipped = n
        break

    per_total = len(passed) + len(failed) + len(errors)

    # If the summary says more tests ran than the per-test regex captured, the
    # regex missed lines — a test emitted stdout so "... ok" wrapped to a new
    # line, or a verbose docstring-style description replaced the
    # "test_x (module.Class)" prefix. Trust the authoritative Ran-N total:
    # failures/errors come from the FAILED(...) summary; the remainder, minus
    # skips and expected failures, are passes.
    if ran_count > per_total:
        failed_n = max(len(failed), sm_failed)
        errors_n = max(len(errors), sm_errors)
        passed_n = max(len(passed),
                       ran_count - failed_n - errors_n - sm_skipped - sm_xfail)
        return {'passed': passed, 'failed': failed, 'errors': errors,
                'passed_count': passed_n, 'failed_count': failed_n,
                'error_count': errors_n}

    # Per-test lines fully account for the run: trust the node-id lists.
    if passed or failed or errors:
        return {'passed': passed, 'failed': failed, 'errors': errors,
                'passed_count': len(passed), 'failed_count': len(failed),
                'error_count': len(errors)}

    # No per-test lines at all (verbosity < 2): derive counts from the summary.
    if ran_count > 0:
        passed_n = max(0, ran_count - sm_failed - sm_errors - sm_skipped - sm_xfail)
        return {'passed': passed, 'failed': failed, 'errors': errors,
                'passed_count': passed_n, 'failed_count': sm_failed,
                'error_count': sm_errors}

    # No useful info found
    return _empty_result()


def _has_signal(result: dict) -> bool:
    """True if a parse result accounts for at least one executed test."""
    return (result['passed_count'] + result['failed_count']
            + result['error_count']) > 0


def parse_run_output(runner: str, output: str) -> dict:
    """Parse test output according to *runner*, with content-based fallback.

    The *runner* (inferred from the command) only picks the PRIMARY parser; if
    it finds nothing we retry with the other parser so a misclassified command
    can't drop a perfectly parseable run on the floor:

      * 'django'      -> ``manage.py test`` emits unittest format; pytest-django
                         emits pytest format. Try unittest first, then pytest.
      * 'unittest'    -> unittest first, then pytest.
      * 'unsupported' -> the command is opaque (e.g. a ``runtests.py`` wrapper or
                         a heredoc that sets up settings then runs the suite), but
                         the OUTPUT still carries a clear pytest/unittest summary.
                         Detect it from the output instead of giving up.
      * 'pytest'/'none' -> pytest first, then unittest.
    """
    if runner in ('django', 'unittest'):
        res = parse_unittest_output(output)
        if not _has_signal(res):
            res = parse_pytest_output(output)
        return res
    if runner == 'unsupported':
        res = parse_pytest_output(output)
        if not _has_signal(res):
            res = parse_unittest_output(output)
        return res
    res = parse_pytest_output(output)
    if not _has_signal(res):
        res = parse_unittest_output(output)
    return res


def compute_transitions(runner: str, before: dict, after: dict) -> dict:
    """Compute FAIL_TO_PASS / PASS_TO_FAIL / PASS_TO_PASS from two parse results.

    FAIL_TO_PASS = tests failing/erroring BEFORE the source patch that PASS after.
    PASS_TO_FAIL = tests passing BEFORE that fail/error after (a regression).
    PASS_TO_PASS = tests passing both before and after.

    For the 'unsupported' runner the command was opaque, but the OUTPUT may now
    be parsed into node ids (e.g. a ``runtests.py`` wrapper emitting pytest
    output). When node ids are present we compute exact transitions like any
    other runner; only when the output yielded NO node ids do we fall back to
    the passed-count delta with empty name lists.

    Returns:
      {
        'fail_to_pass': [...], 'pass_to_fail': [...], 'pass_to_pass': [...],
        'f2p_count': int, 'p2f_count': int, 'p2p_count': int,
      }
    """
    before_failed = set(before['failed']) | set(before['errors'])
    before_passed = set(before['passed'])
    after_passed = set(after['passed'])
    after_failed = set(after['failed']) | set(after['errors'])

    have_node_ids = bool(
        before_failed or before_passed or after_passed or after_failed)

    if runner == 'unsupported' and not have_node_ids:
        f2p_count = max(0, after['passed_count'] - before['passed_count'])
        p2f_count = max(0, before['passed_count'] - after['passed_count'])
        p2p_count = min(before['passed_count'], after['passed_count'])
        return {
            'fail_to_pass': [], 'pass_to_fail': [], 'pass_to_pass': [],
            'f2p_count': f2p_count, 'p2f_count': p2f_count, 'p2p_count': p2p_count,
        }

    fail_to_pass = sorted(before_failed & after_passed)
    pass_to_fail = sorted(before_passed & after_failed)
    pass_to_pass = sorted(before_passed & after_passed)
    return {
        'fail_to_pass': fail_to_pass,
        'pass_to_fail': pass_to_fail,
        'pass_to_pass': pass_to_pass,
        'f2p_count': len(fail_to_pass),
        'p2f_count': len(pass_to_fail),
        'p2p_count': len(pass_to_pass),
    }
