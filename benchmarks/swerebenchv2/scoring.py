"""Resolution scoring shared by the SWE-rebench-V2 judge and evaluator.

Both the inline judge and the offline ``eval_infer.py`` script compare the
tests parsed from a run against the instance's expected ``FAIL_TO_PASS`` /
``PASS_TO_PASS`` lists. A task is resolved when every expected test is reported
as PASSED; tests that were not part of the expectation are ignored, which
matches the standard SWE-bench / SWE-rebench resolution criterion (all
fail-to-pass tests pass and no pass-to-pass test regresses).

The stricter set-equality comparison (``passed == expected``) is used by the
upstream V2 ``scripts/eval.py`` as its ``passed_match`` golden-validation flag;
it is intentionally *not* used for scoring, because an agent that adds its own
passing regression tests would otherwise be marked unresolved.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class TestOutcome:
    """Comparison of parsed PASSED tests against the expected test lists.

    All names are expected to be normalized (timing suffixes stripped) by the
    caller before being passed in.
    """

    passed: tuple[str, ...]
    fail_to_pass_expected: tuple[str, ...]
    pass_to_pass_expected: tuple[str, ...]

    @property
    def expected(self) -> frozenset[str]:
        return frozenset(self.fail_to_pass_expected) | frozenset(
            self.pass_to_pass_expected
        )

    @property
    def resolved(self) -> bool:
        """Whether every expected test was reported as PASSED."""
        return self.expected <= frozenset(self.passed)

    @property
    def fail_to_pass_passed(self) -> tuple[str, ...]:
        passed = frozenset(self.passed)
        return tuple(n for n in self.fail_to_pass_expected if n in passed)

    @property
    def pass_to_pass_failed(self) -> tuple[str, ...]:
        passed = frozenset(self.passed)
        return tuple(n for n in self.pass_to_pass_expected if n not in passed)

    @property
    def missing_expected(self) -> tuple[str, ...]:
        """Expected tests that were not reported as PASSED."""
        passed = frozenset(self.passed)
        return tuple(n for n in sorted(self.expected) if n not in passed)

    @property
    def extra_passed(self) -> tuple[str, ...]:
        """PASSED tests that are not part of the expectation (ignored)."""
        expected = self.expected
        return tuple(n for n in sorted(set(self.passed)) if n not in expected)


def score_tests(
    *,
    passed: Iterable[str],
    fail_to_pass: Iterable[str],
    pass_to_pass: Iterable[str],
) -> TestOutcome:
    """Compare parsed PASSED test names with the expected test names.

    Duplicate names are collapsed and results are sorted so reports are
    deterministic.
    """
    return TestOutcome(
        passed=tuple(sorted(set(passed))),
        fail_to_pass_expected=tuple(sorted(set(fail_to_pass))),
        pass_to_pass_expected=tuple(sorted(set(pass_to_pass))),
    )
