# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from unittest.mock import MagicMock, patch

import pytest

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.smart_client import SmartClient, SmartClientAsync


@pytest.mark.asyncio
async def test_async_smart_client_close() -> None:
    """Test explicit close on Async Client."""
    engine = ArbitrageEngine()
    client = SmartClientAsync(engine)
    await client.close()


def test_sync_smart_client_context_manager() -> None:
    """Test Sync Client as Context Manager."""
    engine = ArbitrageEngine()
    with SmartClient(engine) as client:
        assert isinstance(client, SmartClient)


def test_sync_smart_client_close() -> None:
    """Test explicit close on Sync Client."""
    engine = ArbitrageEngine()
    client = SmartClient(engine)
    client.close()


def test_sync_smart_client_context_manager_full_usage() -> None:
    """Test Sync Client as Context Manager with usage to cover all paths."""
    engine = ArbitrageEngine()
    # Register a model to avoid routing errors
    model = ModelDefinition(
        id="model-a",
        provider="provider-a",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
        is_healthy=True,
    )
    engine.registry.register_model(model)

    with SmartClient(engine) as client:
        # Cover router getter in context
        r = client.chat.completions.router
        assert r is not None

        # Cover router setter in context
        client.chat.completions.router = r

        # Cover create() in context (Portal path)
        with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
            mock_resp = MagicMock()
            mock_resp.usage.prompt_tokens = 0
            mock_resp.usage.completion_tokens = 0
            mock_completion.return_value = mock_resp

            client.chat.completions.create([{"role": "user", "content": "hi"}])

        # Cover close() in context
        client.close()
