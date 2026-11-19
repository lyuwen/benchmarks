from benchmarks.swerebenchv2.scoring import score_tests


def test_extra_passing_tests_do_not_block_resolution() -> None:
    """An agent's own passing regression tests must not make the task fail."""
    outcome = score_tests(
        passed=["TestFix", "TestExisting", "TestAgentOwnRegression"],
        fail_to_pass=(n for n in ["TestFix"]),
        pass_to_pass=["TestExisting"],
    )

    assert outcome.resolved is True
    assert outcome.missing_expected == ()
    assert outcome.extra_passed == ("TestAgentOwnRegression",)
    assert outcome.fail_to_pass_passed == ("TestFix",)
    assert outcome.pass_to_pass_failed == ()


def test_exact_match_resolves() -> None:
    outcome = score_tests(
        passed=["TestExisting", "TestFix"],
        fail_to_pass=["TestFix"],
        pass_to_pass=["TestExisting"],
    )

    assert outcome.resolved is True
    assert outcome.extra_passed == ()
    assert outcome.missing_expected == ()


def test_missing_fail_to_pass_test_does_not_resolve() -> None:
    outcome = score_tests(
        passed=["TestExisting"],
        fail_to_pass=["TestFix"],
        pass_to_pass=["TestExisting"],
    )

    assert outcome.resolved is False
    assert outcome.fail_to_pass_passed == ()
    assert outcome.missing_expected == ("TestFix",)


def test_regressed_pass_to_pass_test_does_not_resolve() -> None:
    outcome = score_tests(
        passed=["TestFix", "TestExisting"],
        fail_to_pass=["TestFix"],
        pass_to_pass=["TestExisting", "TestRegressed"],
    )

    assert outcome.resolved is False
    assert outcome.pass_to_pass_failed == ("TestRegressed",)
    assert outcome.missing_expected == ("TestRegressed",)


def test_duplicate_passed_names_are_collapsed() -> None:
    outcome = score_tests(
        passed=["TestFix", "TestFix", "TestExisting"],
        fail_to_pass=["TestFix", "TestFix"],
        pass_to_pass=["TestExisting"],
    )

    assert outcome.resolved is True
    assert outcome.extra_passed == ()
    assert outcome.fail_to_pass_expected == ("TestFix",)
    assert outcome.passed == ("TestExisting", "TestFix")
