# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from coreason_arbitrage.utils.logger import logger

# Configuration constants
FAILURE_THRESHOLD = 3
FAILURE_WINDOW_SECONDS = 60
COOLDOWN_PERIOD_SECONDS = 300


class LoadBalancer:
    """
    Manages provider health states.
    Implements a Circuit Breaker pattern:
    - Tracks failures within a rolling window (1 minute).
    - If failures > threshold (3), marks provider as unhealthy for a cooldown period (5 minutes).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Stores timestamps of failures for each provider
        self._failures: Dict[str, Deque[float]] = defaultdict(deque)
        # Stores the timestamp when the provider can become healthy again
        self._cooldown_until: Dict[str, float] = {}

    def record_failure(self, provider: str) -> None:
        """
        Records a failure for a specific provider.
        Triggers cooldown if threshold is exceeded.
        """
        with self._lock:
            now = time.time()

            # If already in cooldown, extend? Or just ignore?
            # Standard circuit breaker: usually we don't spam errors if open,
            # but if we are "half-open" or probing, we might record.
            # For simplicity, we just record everything.

            # Prune old failures outside the window
            self._prune_failures(provider, now)

            self._failures[provider].append(now)
            count = len(self._failures[provider])

            logger.warning(f"Recorded failure for {provider}. Count in window: {count}")

            if count > FAILURE_THRESHOLD:
                cooldown_end = now + COOLDOWN_PERIOD_SECONDS
                self._cooldown_until[provider] = cooldown_end
                logger.error(f"Provider {provider} exceeded failure threshold. Marked unhealthy until {cooldown_end}")

    def record_success(self, provider: str) -> None:
        """
        Records a success.
        If a provider was unhealthy (and we are technically testing it), this could reset it.
        For this implementation: A success clears the failure history for the window,
        essentially resetting the circuit breaker if it was closed or half-open.
        However, if it's strictly in cooldown, we might not see successes unless we allow "probing".

        Assumption: If a call succeeds, the provider is healthy.
        """
        with self._lock:
            if provider in self._failures:
                self._failures[provider].clear()

            if provider in self._cooldown_until:
                del self._cooldown_until[provider]
                logger.info(f"Provider {provider} recovered and marked healthy.")

    def is_provider_healthy(self, provider: str) -> bool:
        """
        Checks if the provider is currently healthy (not in cooldown).
        """
        with self._lock:
            now = time.time()

            # Check cooldown
            if provider in self._cooldown_until:
                if now < self._cooldown_until[provider]:
                    return False
                else:
                    # Cooldown expired
                    del self._cooldown_until[provider]
                    # Note: We don't clear failures automatically here,
                    # they will age out naturally or be cleared on success.
                    logger.info(f"Provider {provider} cooldown expired. Marked healthy.")
                    return True

            return True

    def _prune_failures(self, provider: str, now: float) -> None:
        """
        Removes failure timestamps that are older than the window.
        """
        timestamps = self._failures[provider]
        while timestamps and (now - timestamps[0] > FAILURE_WINDOW_SECONDS):
            timestamps.popleft()
