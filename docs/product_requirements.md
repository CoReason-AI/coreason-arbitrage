# Product Requirements Document: coreason-arbitrage

Domain: Model Routing, Cost Optimization, & FinOps
Architectural Role: The "Traffic Controller" / The Smart Switch
Core Philosophy: "The Right Model for the Right Task. Don't use a PhD to do arithmetic."
Dependencies: coreason-budget (Quotas), coreason-veritas (Audit), coreason-model-foundry (Custom Models), litellm (Provider Abstraction)

## ---

**1. Executive Summary**

coreason-arbitrage is the intelligent routing layer that sits between the Agent and the Model Providers. It prevents "Token Burn" by enforcing a **Cascading Model Strategy**.

Instead of hardcoding "Use GPT-4," the Cortex requests "A Model capable of High Reasoning." Arbitrage then evaluates the prompt, checks the user's remaining budget, checks the health of downstream APIs, and returns a client for the optimal model (e.g., a fine-tuned Llama-3 or Claude 3.5 Sonnet). It also acts as a **Circuit Breaker**, automatically failing over to backup providers (Azure $\to$ AWS) during outages.

## **2. Functional Philosophy**

The agent must implement the **Classify-Route-Fallback-Log Loop**:

1. **Complexity Analysis:** Before routing, we run the prompt through a lightweight classifier (DistilBERT or RegEx rules). Is this a "Hello" (Complexity: 0.1) or a "Clinical Protocol Analysis" (Complexity: 0.9)?
2. **The Waterfall (Cascading):**
   * *Tier 1 (Fast/Cheap):* Llama-3-8B / Haiku. Used for extraction, formatting, simple Q&A.
   * *Tier 2 (Smart/Mid):* Llama-3-70B / Sonnet. Used for drafting, summarization.
   * *Tier 3 (Genius/Expensive):* GPT-4o / Opus / o1. Used only for complex reasoning and final verification.
3. **Provider Agnosticism:** The routing logic is decoupled from the vendor. We route to "High Intelligence," not "OpenAI." This prevents vendor lock-in.
4. **Resiliency:** If Azure OpenAI returns a 429 (Rate Limit), Arbitrage instantly retries on AWS Bedrock without the user noticing.

## ---

**3. Core Functional Requirements (Component Level)**

### **3.1 The Gatekeeper (The Classifier)**

**Concept:** A millisecond-latency pre-processor.

* **Mechanism:**
  * **Heuristic Check:** Length > 2000 chars? Contains keywords "Analyze", "Critique", "Reason"? $\to$ High Complexity.
  * **Model Check (Optional):** Runs a tiny, quantized BERT model to predict "Difficulty Score" (0.0 - 1.0).
* **Output:** RoutingContext(complexity=0.85, domain="medical").

### **3.2 The Router (The Switch)**

**Concept:** The decision engine.

* **Logic:**
  * *Input:* RoutingContext, UserBudget.
  * *Rule:* IF complexity > 0.8 OR domain == "safety_critical" THEN use Tier_3 ELSE use Tier_1.
  * *Optimization:* If UserBudget is low (< 10%), downgrade non-critical requests to Tier 1 automatically ("Economy Mode").
* **Custom Model Priority:** If coreason-model-foundry has registered a specialized model for this domain (e.g., "Oncology-Llama-3"), prefer that over generic models.

### **3.3 The Load Balancer (The Failover)**

**Concept:** Ensures 99.99% availability.

* **Mechanism:**
  * Maintains a "Health Score" for each provider endpoint.
  * If Azure returns 5xx/429 errors > 3 times in 1 minute, mark Azure as "Unhealthy" for 5 minutes.
  * **Failover:** Route traffic to the secondary configured provider (e.g., AWS Bedrock or Self-Hosted vLLM).

### **3.4 The Accountant (The Ledger)**

**Concept:** Real-time cost tracking.

* **Action:**
  * Calculates cost *after* the response is received (Input Tokens + Output Tokens).
  * Deducts from coreason-budget.
  * Logs the precise cost and model used to coreason-veritas.

## ---

**4. Integration Requirements**

* **coreason-cortex:**
  * Cortex does *not* import OpenAI directly. It imports arbitrage.
  * Code: llm = arbitrage.get_client(capability="reasoning").
* **coreason-model-foundry:**
  * When Foundry finishes training, it registers the new model ID in Arbitrage's registry: "I am ready to handle Oncology queries."
* **coreason-budget:**
  * Arbitrage queries budget.check_allowance(user_id) before every call.

## ---

**5. User Stories**

### **Story A: The "Simple Greeting" (Cost Save)**

User Prompt: "Hi, are you there?"
Gatekeeper: Complexity Score = 0.05.
Router: Selects Tier 1 (Llama-3-8B).
Result: Response generated for $0.000001. User is happy. (GPT-4 would have cost 100x more).

### **Story B: The "Complex Protocol" (Intelligence Upgrade)**

User Prompt: "Analyze this attached PDF for exclusion criteria conflicts."
Gatekeeper: Complexity Score = 0.95. Domain = "Clinical".
Router: Selects Tier 3 (GPT-4o or Claude Opus).
Result: High-fidelity reasoning. Cost is justified by value.

### **Story C: The "Azure Outage" (Resiliency)**

Context: Azure US-East is down.
Router: Attempts to call Tier 3 model on Azure. Receives 503 Service Unavailable.
Load Balancer: Detects failure. Instantly switches route to AWS Bedrock (Claude 3.5 Sonnet).
Result: User gets a response. They never knew Azure was down.

## ---

**6. Data Schema**

### **RoutingPolicy (YAML)**

YAML

policies:
  - name: "safety_critical"
    condition: "complexity >= 0.8 or 'adverse event' in prompt"
    models: ["gpt-4o", "claude-3-opus"]
    fallback: ["llama-3-70b-instruct"]

  - name: "general_chat"
    condition: "default"
    models: ["llama-3-8b-instruct", "gpt-3.5-turbo"]
    fallback: ["mistral-7b"]

### **ModelRegistry (Dynamic State)**

Python

class ModelTier(str, Enum):
    TIER_1_FAST = "fast"
    TIER_2_SMART = "smart"
    TIER_3_REASONING = "reasoning"

class ModelDefinition(BaseModel):
    id: str                  # "azure/gpt-4o"
    provider: str            # "azure"
    tier: ModelTier
    cost_per_1k_input: float
    cost_per_1k_output: float
    is_healthy: bool = True

## ---

**7. Implementation Directives for the Coding Agent**

1. **Use litellm:** Do NOT write custom API wrappers for OpenAI, Azure, Bedrock, etc. Use the **litellm** library. It provides a unified interface (completion(model="azure/gpt-4", ...)) and handles exceptions/retries automatically.
2. **Singleton Pattern:** The ArbitrageEngine must be a singleton to maintain the state of "Provider Health" across requests (e.g., remembering that Azure is down).
3. **Latency Budget:** The classification step must be fast. Do not use an LLM to classify an LLM prompt (infinite recursion/cost). Use regex, length heuristics, or a tiny local ONNX model.
4. **Fail Open vs. Closed:**
   * If the *Router* crashes, **Fail Open** to a safe default (e.g., GPT-4o) so the user is not blocked, but log a CRITICAL error.
   * If the *Budget* check fails (DB down), **Fail Closed** (Deny request) to prevent unlimited spending.

## **Final Prompt Instructions**

### **Mandatory Code Quality & CI Rules**

You **must** strictly follow this workflow before producing the final answer:

1. **After all code changes are made, run:**

    `ruff format .`

`ruff check --fix .`

       **2. Then run full pre-commit validation:**

	`pre-commit run --all-files`

    `3.` **If ANY files are modified by these commands:**

* You **must** stage and commit those changes.

  * Then repeat step 2.

**4. The task is NOT complete until:**

	`pre-commit run --all-files`

 finishes with:

* **No file modifications**

  * **No hook failures**

  * **No mypy errors**

  5. The final branch **must** pass all pre-commit hooks without making changes.
