# Usage Guide

`coreason-arbitrage` operates in two modes: as a **Python Library** imported directly into your Cortex, or as a standalone **Smart Routing Microservice**.

## 1. Library Usage

Import the engine directly to gain fine-grained control over routing within your application code.

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

## 2. Server Mode (Microservice)

Run `coreason-arbitrage` as a containerized service to act as a central traffic controller for your platform.

### Running via Docker

The official Docker image comes pre-configured with `uvicorn`.

```bash
docker run -p 8000:8000 coreason/arbitrage:v0.3.0
```

### API Endpoints

Once running, the service exposes an OpenAI-compatible API.

#### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{"status": "ready", "routing_engine": "active"}
```

#### Chat Completions

Route a prompt through the arbitrage engine. The service handles classification, routing, and failover automatically.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "Analyze this clinical trial protocol."}
    ],
    "user": "user_123"
  }'
```

**Note:** The `model` parameter is optional and often overridden by the Router's intelligent selection logic (e.g., upgrading to Tier 3 for complex queries).
