import re
from typing import Dict, List, Optional

from coreason_arbitrage.models import RoutingContext
from coreason_arbitrage.utils.logger import logger

COMPLEXITY_THRESHOLD_LENGTH = 2000
# Regex pattern to match whole words only, case-insensitive
COMPLEXITY_KEYWORDS_PATTERN = re.compile(r"\b(analyze|critique|reason)\b", re.IGNORECASE)

# Domain keyword mappings
DOMAIN_MAPPINGS: Dict[str, List[str]] = {
    "safety_critical": ["hazard", "emergency", "danger", "immediate", "adverse event"],
    "medical": ["clinical", "dose"],
}


def _compile_domain_patterns() -> Dict[str, re.Pattern[str]]:
    """
    Compiles regex patterns for each domain using word boundaries.
    """
    patterns = {}
    for domain, keywords in DOMAIN_MAPPINGS.items():
        # Escape keywords to be safe, then join them with OR
        # Note: We need to handle multi-word keywords like "adverse event".
        # \b(foo bar|baz)\b works if "foo" and "bar" are words.
        # However, \b matches between \w and \W.
        # "adverse event" has a space. "adverse" starts/ends with \b. "event" starts/ends with \b.
        # But \b(adverse event)\b matches " adverse event ".
        # Let's ensure keywords are properly escaped.
        escaped_keywords = [re.escape(k) for k in keywords]
        pattern_str = r"\b(" + "|".join(escaped_keywords) + r")\b"
        patterns[domain] = re.compile(pattern_str, re.IGNORECASE)
    return patterns


DOMAIN_PATTERNS = _compile_domain_patterns()


class Gatekeeper:
    """
    The Gatekeeper analyzes the complexity of a prompt to determine routing.
    It uses a lightweight heuristic approach to avoid adding latency.
    """

    def classify(self, text: str) -> RoutingContext:
        """
        Analyzes the input text and returns a RoutingContext with a complexity score
        and an optional domain.

        Heuristics:
        - Complexity 0.9 (High) if:
            - Length > 2000 characters
            - Contains keywords: 'Analyze', 'Critique', 'Reason' (whole words only)
        - Complexity 0.1 (Low) otherwise.

        Domain Extraction:
        - Scans for keywords defined in DOMAIN_MAPPINGS.
        - Returns the first matching domain.
        """
        # Complexity Check
        is_long = len(text) > COMPLEXITY_THRESHOLD_LENGTH
        has_complexity_keywords = bool(COMPLEXITY_KEYWORDS_PATTERN.search(text))

        if is_long or has_complexity_keywords:
            complexity = 0.9
        else:
            complexity = 0.1

        # Domain Check
        detected_domain: Optional[str] = None
        for domain, pattern in DOMAIN_PATTERNS.items():
            if pattern.search(text):
                detected_domain = domain
                break  # Return first match

        logger.debug(
            f"Gatekeeper classification: length={len(text)}, "
            f"has_keywords={has_complexity_keywords} -> complexity={complexity}, "
            f"domain={detected_domain}"
        )

        return RoutingContext(complexity=complexity, domain=detected_domain)
