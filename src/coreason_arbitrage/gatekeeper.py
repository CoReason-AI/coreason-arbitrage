import re

from coreason_arbitrage.models import RoutingContext
from coreason_arbitrage.utils.logger import logger

COMPLEXITY_THRESHOLD_LENGTH = 2000
# Regex pattern to match whole words only, case-insensitive
COMPLEXITY_KEYWORDS_PATTERN = re.compile(r"\b(analyze|critique|reason)\b", re.IGNORECASE)


class Gatekeeper:
    """
    The Gatekeeper analyzes the complexity of a prompt to determine routing.
    It uses a lightweight heuristic approach to avoid adding latency.
    """

    def classify(self, text: str) -> RoutingContext:
        """
        Analyzes the input text and returns a RoutingContext with a complexity score.

        Heuristics:
        - Complexity 0.9 (High) if:
            - Length > 2000 characters
            - Contains keywords: 'Analyze', 'Critique', 'Reason' (whole words only)
        - Complexity 0.1 (Low) otherwise.
        """
        is_long = len(text) > COMPLEXITY_THRESHOLD_LENGTH
        has_keywords = bool(COMPLEXITY_KEYWORDS_PATTERN.search(text))

        if is_long or has_keywords:
            complexity = 0.9
        else:
            complexity = 0.1

        logger.debug(
            f"Gatekeeper classification: length={len(text)}, has_keywords={has_keywords} -> complexity={complexity}"
        )

        return RoutingContext(complexity=complexity, domain=None)
