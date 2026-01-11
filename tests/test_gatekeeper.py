import pytest

from coreason_arbitrage.gatekeeper import COMPLEXITY_THRESHOLD_LENGTH, Gatekeeper


@pytest.fixture
def gatekeeper() -> Gatekeeper:
    return Gatekeeper()


def test_classify_simple_greeting(gatekeeper: Gatekeeper) -> None:
    """Test Story A: The 'Simple Greeting'."""
    text = "Hi, are you there?"
    context = gatekeeper.classify(text)
    assert context.complexity == 0.1
    assert context.domain is None


def test_classify_complex_keyword(gatekeeper: Gatekeeper) -> None:
    """Test Story B: The 'Complex Protocol' via keyword."""
    text = "Please analyze this attached PDF."
    context = gatekeeper.classify(text)
    assert context.complexity == 0.9


def test_classify_complex_keyword_case_insensitive(gatekeeper: Gatekeeper) -> None:
    """Ensure keywords are detected regardless of case."""
    text = "Please CRITIQUE this logic."
    context = gatekeeper.classify(text)
    assert context.complexity == 0.9


def test_classify_long_text(gatekeeper: Gatekeeper) -> None:
    """Test length heuristic."""
    # Create text slightly longer than threshold
    text = "a" * (COMPLEXITY_THRESHOLD_LENGTH + 1)
    context = gatekeeper.classify(text)
    assert context.complexity == 0.9


def test_classify_boundary_length(gatekeeper: Gatekeeper) -> None:
    """Test exact threshold length (should be low complexity)."""
    text = "a" * COMPLEXITY_THRESHOLD_LENGTH
    context = gatekeeper.classify(text)
    assert context.complexity == 0.1


def test_classify_empty_string(gatekeeper: Gatekeeper) -> None:
    """Test empty string input."""
    context = gatekeeper.classify("")
    assert context.complexity == 0.1


def test_classify_false_positive_keywords(gatekeeper: Gatekeeper) -> None:
    """Test that partial matches (e.g. 'reasonable') do not trigger high complexity."""
    text = "This is a reasonable request."
    context = gatekeeper.classify(text)
    assert context.complexity == 0.1


def test_classify_false_positive_keywords_2(gatekeeper: Gatekeeper) -> None:
    """Test that partial matches (e.g. 'breathalyze') do not trigger high complexity."""
    text = "We should breathalyze the driver."
    context = gatekeeper.classify(text)
    assert context.complexity == 0.1


# --- Edge Cases & Complex Scenarios ---


def test_classify_punctuation_boundaries(gatekeeper: Gatekeeper) -> None:
    """Test keywords followed immediately by punctuation."""
    texts = ["Please analyze.", "Why do you reason?", "Critique: This is a test."]
    for text in texts:
        context = gatekeeper.classify(text)
        assert context.complexity == 0.9, f"Failed to detect keyword in: '{text}'"


def test_classify_code_variable_names(gatekeeper: Gatekeeper) -> None:
    """
    Test that keywords embedded in snake_case or camelCase variables
    do NOT trigger high complexity (false positives).
    The regex \b should treat '_' as a word character, preventing a match.
    """
    texts = ["def analyze_data(self): pass", "var_reason = 10", "class CritiqueController:", "analyzeData = True"]
    for text in texts:
        context = gatekeeper.classify(text)
        assert context.complexity == 0.1, f"Incorrectly flagged code variable in: '{text}'"


def test_classify_hyphenated_words(gatekeeper: Gatekeeper) -> None:
    """
    Test hyphenated words.
    Hyphens are usually non-word characters, so 'self-analyze'
    should contain the word 'analyze'.
    """
    text = "Please self-analyze your behavior."
    context = gatekeeper.classify(text)
    assert context.complexity == 0.9


def test_classify_multiline_text(gatekeeper: Gatekeeper) -> None:
    """Test keywords hidden inside multiline text."""
    text = """
    Here is a long prompt.
    I want you to
    analyze
    this data carefully.
    """
    context = gatekeeper.classify(text)
    assert context.complexity == 0.9


def test_classify_strict_keyword_matching(gatekeeper: Gatekeeper) -> None:
    """
    Verify strict adherence to keywords.
    'Analyzed' (past tense) is NOT in the list ['analyze', 'critique', 'reason'].
    """
    text = "I have analyzed the data already."
    context = gatekeeper.classify(text)
    assert context.complexity == 0.1
