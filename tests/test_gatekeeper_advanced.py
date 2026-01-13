import pytest

from coreason_arbitrage.gatekeeper import COMPLEXITY_THRESHOLD_LENGTH, Gatekeeper


@pytest.fixture
def gatekeeper() -> Gatekeeper:
    return Gatekeeper()


def test_multiple_domains_precedence(gatekeeper: Gatekeeper) -> None:
    """
    Test scenario where keywords from multiple domains appear in the text.
    'safety_critical' is defined before 'medical' in DOMAIN_MAPPINGS (prioritized).
    Text: "The clinical trial reported an immediate danger."
    - 'clinical' -> medical
    - 'danger' -> safety_critical
    Expected: 'safety_critical' (safety checks must take precedence)
    """
    text = "The clinical trial reported an immediate danger."
    context = gatekeeper.classify(text)

    # We expect 'safety_critical' because it is now checked first
    assert context.domain == "safety_critical"
    assert context.complexity == 0.1  # 'danger' is not a complexity keyword, nor 'clinical'


def test_compound_word_exclusion(gatekeeper: Gatekeeper) -> None:
    """
    Test that keywords forming part of a larger word (compound) are NOT matched.
    'hazard' is a keyword. 'biohazard' should NOT trigger it due to \b boundary.
    """
    text = "The container is labeled as biohazard material."
    context = gatekeeper.classify(text)

    # Should NOT match 'safety_critical' via 'hazard'
    assert context.domain is None


def test_hyphenated_word_inclusion(gatekeeper: Gatekeeper) -> None:
    """
    Test that keywords connected by hyphens ARE matched.
    Hyphens are non-word characters, so boundaries should work.
    'clinical' is a keyword. 'non-clinical' should trigger it.
    """
    text = "This is a non-clinical study."
    context = gatekeeper.classify(text)

    assert context.domain == "medical"


def test_keyword_at_end_of_long_text(gatekeeper: Gatekeeper) -> None:
    """
    Test detecting a domain keyword at the very end of a text.
    Also tests that complexity is High due to length, AND domain is detected.
    """
    # Create long text padding
    padding = "word " * (COMPLEXITY_THRESHOLD_LENGTH // 5 + 1)
    text = padding + " unexpected adverse event"

    context = gatekeeper.classify(text)

    # Should be High complexity due to length
    assert context.complexity == 0.9
    # Should be 'safety_critical' due to 'adverse event'
    assert context.domain == "safety_critical"


def test_json_structure_analysis(gatekeeper: Gatekeeper) -> None:
    """
    Test scanning within a JSON-like string.
    Keywords in keys or values should be detected if they are whole words.
    """
    text = '{"reason": "checking for clinical data", "value": 10}'
    context = gatekeeper.classify(text)

    # "reason" is a complexity keyword -> 0.9
    # "clinical" is a domain keyword -> medical
    assert context.complexity == 0.9
    assert context.domain == "medical"


def test_json_key_false_positive(gatekeeper: Gatekeeper) -> None:
    """
    Test that 'danger_level' (snake_case) does not trigger 'danger'.
    """
    text = '{"danger_level": "high", "status": "ok"}'
    context = gatekeeper.classify(text)

    # "danger_level" contains "danger" but underscore is a word char,
    # so \bdanger\b should NOT match.
    assert context.domain is None
