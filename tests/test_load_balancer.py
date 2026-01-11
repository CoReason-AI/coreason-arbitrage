from unittest.mock import patch

import pytest

from coreason_arbitrage.load_balancer import (
    COOLDOWN_PERIOD_SECONDS,
    FAILURE_THRESHOLD,
    FAILURE_WINDOW_SECONDS,
    LoadBalancer,
)


@pytest.fixture
def lb() -> LoadBalancer:
    return LoadBalancer()


def test_initial_state(lb: LoadBalancer) -> None:
    assert lb.is_provider_healthy("azure") is True
    assert lb.is_provider_healthy("aws") is True


def test_failure_accumulation(lb: LoadBalancer) -> None:
    provider = "azure"

    # Record failures up to threshold
    for _ in range(FAILURE_THRESHOLD):
        lb.record_failure(provider)
        assert lb.is_provider_healthy(provider) is True

    # Record one more failure to trip the breaker
    lb.record_failure(provider)
    assert lb.is_provider_healthy(provider) is False


def test_failure_window_expiry(lb: LoadBalancer) -> None:
    provider = "azure"

    with patch("time.time") as mock_time:
        start_time = 1000.0
        mock_time.return_value = start_time

        # Add failures
        for _ in range(FAILURE_THRESHOLD):
            lb.record_failure(provider)

        # Advance time past the window
        mock_time.return_value = start_time + FAILURE_WINDOW_SECONDS + 1

        # Add one more failure.
        # Since previous ones expired, total count should be 1, so still healthy.
        lb.record_failure(provider)
        assert lb.is_provider_healthy(provider) is True


def test_cooldown_period(lb: LoadBalancer) -> None:
    provider = "azure"

    with patch("time.time") as mock_time:
        start_time = 1000.0
        mock_time.return_value = start_time

        # Trip the breaker
        for _ in range(FAILURE_THRESHOLD + 1):
            lb.record_failure(provider)

        assert lb.is_provider_healthy(provider) is False

        # Advance time within cooldown
        mock_time.return_value = start_time + COOLDOWN_PERIOD_SECONDS - 1
        assert lb.is_provider_healthy(provider) is False

        # Advance time past cooldown
        mock_time.return_value = start_time + COOLDOWN_PERIOD_SECONDS + 1
        assert lb.is_provider_healthy(provider) is True


def test_recovery_on_success(lb: LoadBalancer) -> None:
    provider = "azure"

    # Trip the breaker
    for _ in range(FAILURE_THRESHOLD + 1):
        lb.record_failure(provider)

    assert lb.is_provider_healthy(provider) is False

    # Record success
    lb.record_success(provider)

    assert lb.is_provider_healthy(provider) is True


def test_multiple_providers(lb: LoadBalancer) -> None:
    lb.record_failure("azure")
    # Trip aws
    for _ in range(FAILURE_THRESHOLD + 1):
        lb.record_failure("aws")

    assert lb.is_provider_healthy("azure") is True
    assert lb.is_provider_healthy("aws") is False
