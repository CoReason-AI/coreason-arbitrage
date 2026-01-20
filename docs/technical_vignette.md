# The Architecture and Utility of coreason-arbitrage

### 1. The Philosophy (The Why)

In the rapidly evolving landscape of Large Language Models (LLMs), a critical inefficiency has emerged: **Token Burn**. Developers often default to the most capable (and expensive) models—like GPT-4 or Claude 3 Opus—for tasks that require only a fraction of their intelligence. Conversely, using a smaller model for complex reasoning leads to failure and retry loops, which are equally costly.

`coreason-arbitrage` was built to solve this "Goldilocks Problem." It acts as an intelligent **Traffic Controller** sitting between your application and the model providers. Its philosophy is simple: **"The Right Model for the Right Task."**

By decoupling the application logic from specific model providers, `coreason-arbitrage` prevents vendor lock-in. It treats intelligence as a commodity, routing requests to the most cost-effective provider that meets the task's complexity requirements. Furthermore, it introduces robust failover mechanisms, ensuring that if one provider goes down (e.g., an Azure outage), your application seamlessly switches to another (e.g., AWS Bedrock) without the user ever noticing.

### 2. Under the Hood (The Dependencies & Logic)

The package leverages a focused stack to deliver high-performance routing and reliability:

*   **`litellm`**: This is the engine room of provider abstraction. It allows `coreason-arbitrage` to communicate with virtually any LLM provider (OpenAI, Azure, Anthropic, Bedrock, etc.) using a unified interface. This eliminates the need for maintaining multiple API wrappers.
*   **`pydantic`**: Used for rigorous data validation and schema definition. The `RoutingContext`, `ModelDefinition`, and configuration objects ensure that data flowing through the system is structured and predictable.
*   **`coreason-*` ecosystem**: The package is designed as a modular component within a larger system. It defines strict `Protocol` interfaces for:
    *   **Budgeting (`BudgetClient`)**: Enforcing quotas and "Economy Mode."
    *   **Auditing (`AuditClient`)**: Logging transactions for compliance and cost analysis.
    *   **Model Foundry (`ModelFoundryClient`)**: discovering custom, fine-tuned models.

**The Logic Flow:**
The core architectural pattern is the **Classify-Route-Fallback loop**:
1.  **Gatekeeper**: A low-latency classifier analyzes the incoming prompt's complexity (e.g., "Is this a simple greeting or a clinical protocol analysis?").
2.  **Router**: Based on the complexity score, user budget, and domain (e.g., "medical"), the Router selects the optimal model tier. It prioritizes specialized models over generic ones.
3.  **Load Balancer**: Before finalizing the selection, the Load Balancer checks the health of the chosen provider. If a provider is experiencing high error rates, it is automatically excluded.
4.  **Smart Execution**: The request is sent. If it fails due to a transient error (like a rate limit), the system retries, potentially failing over to a backup provider in real-time.

### 3. In Practice (The How)

Using `coreason-arbitrage` is designed to feel like using a standard OpenAI client, but with "superpowers."

**Example 1: The Smart Client Setup**
Instead of hardcoding a model, you request a client with a specific capability. The `ArbitrageEngine` handles the complexity behind the scenes.

```python
from coreason_arbitrage.engine import ArbitrageEngine

# Initialize the engine (typically done once as a singleton)
engine = ArbitrageEngine()

# Configure with your ecosystem clients (Budget, Audit, etc.)
# engine.configure(budget_client=..., audit_client=..., foundry_client=...)

# Get a "Smart Client" - looks like OpenAI, acts like a Traffic Controller
client = engine.get_client(capability="reasoning")
```

**Example 2: Automatic Routing and Failover**
When you make a call, the `Gatekeeper` analyzes your prompt. If you send a complex medical query, it routes to a high-tier model. If the primary provider fails, it seamlessly switches.

```python
# A complex query that triggers Tier 3 (Reasoning) models
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Analyze the attached clinical protocol for exclusion criteria conflicts."}
    ],
    user="dr_house"  # Used for budget and audit tracking
)

print(response.choices[0].message.content)
```

In this scenario, `coreason-arbitrage` might have:
1.  Detected high complexity (> 0.8).
2.  Identified the "clinical" domain.
3.  Attempted to use `azure/gpt-4o`.
4.  Encountered a Rate Limit error.
5.  Automatically failed over to `aws/claude-3-opus`.
6.  Returned the response to you—all in a single function call.
