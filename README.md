# coreason-arbitrage

The "Traffic Controller" / The Smart Switch for CoReason-AI.

[![Organization](https://img.shields.io/badge/org-CoReason--AI-blue)](https://github.com/CoReason-AI)
[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://github.com/CoReason-AI/coreason-arbitrage/blob/main/LICENSE)
[![Build Status](https://github.com/CoReason-AI/coreason-arbitrage/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason-arbitrage/actions)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-Product%20Requirements-informational)](docs/product_requirements.md)

## Installation

```bash
pip install coreason-arbitrage
```

## Features

*   **Intelligent Routing:** Cascading model strategy to select the right model for the right task (Tier 1, Tier 2, Tier 3).
*   **Cost Optimization:** Prevents "Token Burn" by routing simpler tasks to cheaper models and using Economy Mode.
*   **Resiliency:** Circuit Breaker mechanism to failover to backup providers during outages.
*   **Provider Agnosticism:** Decoupled from specific vendors, avoiding lock-in.
*   **FinOps:** Real-time cost tracking and logging.

See [Product Requirements](docs/product_requirements.md) for more details.

## Usage

```python
from coreason_arbitrage.engine import ArbitrageEngine

# Initialize the engine (Singleton)
# Note: You can optionally configure it with your specific clients
engine = ArbitrageEngine()

# Get a smart client capable of handling the request
# This client mimics the OpenAI interface but routes intelligently
client = engine.get_client()

# Use the client to create a completion
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
    # Arbitrage routing happens automatically based on content analysis
)

print(response.choices[0].message.content)
```
