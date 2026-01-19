# coreason-arbitrage

[![License: Prosperity 3.0](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![CI](https://github.com/CoReason-AI/coreason_arbitrage/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_arbitrage/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**coreason-arbitrage** is the intelligent routing layer ("Traffic Controller") for CoReason-AI. It sits between agents and model providers to optimize cost, performance, and reliability.

Instead of hardcoding specific models, it dynamically selects the "Right Model for the Right Task" based on prompt complexity, domain context, provider health, and budget constraints.

## Features

-   **Cascading Model Strategy:** Automatically routes requests to the most appropriate tier:
    -   *Tier 1 (Fast/Cheap):* Simple extraction, formatting (e.g., Llama-3-8B).
    -   *Tier 2 (Smart/Mid):* Drafting, summarization (e.g., Llama-3-70B).
    -   *Tier 3 (Genius/Expensive):* Complex reasoning, safety-critical tasks (e.g., GPT-4o).
-   **Gatekeeper:** Millisecond-latency classifier (Heuristic/Regex) to determine prompt complexity and domain.
-   **Circuit Breaker & Failover:** Automatically detects provider outages (e.g., Azure 5xx/429) and seamlessly fails over to backup providers (e.g., AWS Bedrock) without user interruption.
-   **Economy Mode:** Downgrades non-critical requests to cheaper tiers if the user's budget is running low (<10%).
-   **Provider Agnostic:** Built on top of [litellm](https://github.com/BerriAI/litellm) to support 100+ LLMs.
-   **Fail-Open Design:** Prioritizes system availability, falling back to safe defaults if components fail.

## Installation

```bash
pip install coreason-arbitrage
```

## Usage

```python
import os
from coreason_arbitrage import ArbitrageEngine

# Ensure environment variables for providers are set
# os.environ["AZURE_API_KEY"] = "..."

# Initialize the engine (Singleton)
# Ideally, configure it with Budget and Audit clients in a production setup
engine = ArbitrageEngine()

# Get a Smart Client
# This client mimics the OpenAI Python client interface
client = engine.get_client(capability="reasoning")

# Make a request
# The 'model' parameter is handled by the Router based on the prompt
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Analyze the attached clinical protocol for adverse events."}
    ],
    user="user_123"
)

print(response.choices[0].message.content)
```

## Configuration

The `ArbitrageEngine` can be configured with external dependencies for full functionality:

```python
from coreason_arbitrage.interfaces import BudgetClient, AuditClient, ModelFoundryClient

class MyBudgetClient(BudgetClient):
    ...

# engine.configure(budget_client=..., audit_client=..., foundry_client=...)
```

## License

This software is licensed under the **Prosperity Public License 3.0**.
See [LICENSE](LICENSE) for more details.
