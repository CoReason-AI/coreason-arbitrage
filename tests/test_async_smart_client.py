from unittest.mock import MagicMock, patch

import pytest

from coreason_arbitrage.engine import ArbitrageEngine
from coreason_arbitrage.models import ModelDefinition, ModelTier
from coreason_arbitrage.smart_client import SmartClientAsync


@pytest.mark.asyncio
async def test_async_smart_client_lifecycle() -> None:
    engine = ArbitrageEngine()
    async with SmartClientAsync(engine) as client:
        assert client._client is not None
        assert hasattr(client.chat.completions, "create")


@pytest.mark.asyncio
async def test_async_smart_client_create_mocked() -> None:
    engine = ArbitrageEngine()
    # Ensure registry has a model to avoid routing error
    engine.registry.clear()
    model = ModelDefinition(
        id="model-a",
        provider="provider-a",
        tier=ModelTier.TIER_1_FAST,
        cost_per_1k_input=0,
        cost_per_1k_output=0,
        is_healthy=True,
    )
    engine.registry.register_model(model)

    async with SmartClientAsync(engine) as client:
        with patch("coreason_arbitrage.smart_client.acompletion") as mock_completion:
            mock_response = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 10
            mock_completion.return_value = mock_response

            # Patch Gatekeeper to be deterministic (optional, but good practice)
            # Just running it is fine as it uses regex.

            response = await client.chat.completions.create([{"role": "user", "content": "hi"}])
            assert response == mock_response
            mock_completion.assert_called_once()
