import pytest

from coreason_arbitrage.gatekeeper import Gatekeeper


@pytest.fixture
def gatekeeper() -> Gatekeeper:
    return Gatekeeper()


def test_mixed_domain_priority(gatekeeper: Gatekeeper) -> None:
    """
    Test that 'safety_critical' takes precedence over 'medical'
    when keywords from both are present.

    'clinical' -> medical
    'adverse event' -> safety_critical

    If safety_critical is not prioritized, this might return 'medical',
    potentially leading to unsafe routing (Tier 1 instead of Tier 3).
    """
    text = "The clinical report indicates an adverse event."
    context = gatekeeper.classify(text)
    assert context.domain == "safety_critical"


def test_negation_fail_safe(gatekeeper: Gatekeeper) -> None:
    """
    Test that even negated safety keywords trigger safety domain.
    'No adverse event' should still be treated with caution (or at least detected).
    """
    text = "There was no adverse event reported."
    context = gatekeeper.classify(text)
    assert context.domain == "safety_critical"


def test_punctuation_handling(gatekeeper: Gatekeeper) -> None:
    """Test that punctuation doesn't break multi-word keyword matching."""
    text = "We observed an adverse event."
    context = gatekeeper.classify(text)
    assert context.domain == "safety_critical"
