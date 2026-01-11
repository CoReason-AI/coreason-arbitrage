from typing import Optional

from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.load_balancer import LoadBalancer
from coreason_arbitrage.models import ModelDefinition, ModelTier, RoutingContext
from coreason_arbitrage.registry import ModelRegistry
from coreason_arbitrage.utils.logger import logger


class Router:
    """
    The Router determines the best model to use based on the routing context,
    user budget, and available models in the registry.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        budget_client: BudgetClient,
        load_balancer: Optional[LoadBalancer] = None,
    ) -> None:
        self.registry = registry
        self.budget_client = budget_client
        self.load_balancer = load_balancer

    def route(self, context: RoutingContext, user_id: str) -> ModelDefinition:
        """
        Selects the optimal model for the given context and user.

        Logic:
        1. Determine Baseline Tier:
           - Tier 3: Complexity >= 0.8 OR Domain == 'safety_critical'
           - Tier 2: 0.4 <= Complexity < 0.8
           - Tier 1: Complexity < 0.4
        2. Apply Economy Mode:
           - If user budget < 10% (0.1) AND Tier is Tier 2, downgrade to Tier 1.
        3. Select Model:
           - Query Registry for the target Tier.
           - Filter by provider health (LoadBalancer).
           - Return the first healthy model.
           - If no model found in target Tier, try fallback (logic TBD, for now raise Error).
        """
        # 1. Determine Baseline Tier
        target_tier: ModelTier

        # Normalize domain for case-insensitive check
        domain_lower = context.domain.lower() if context.domain else ""

        if context.complexity >= 0.8 or domain_lower == "safety_critical":
            target_tier = ModelTier.TIER_3_REASONING
        elif context.complexity >= 0.4:
            target_tier = ModelTier.TIER_2_SMART
        else:
            target_tier = ModelTier.TIER_1_FAST

        logger.debug(
            f"Baseline Tier selection: {target_tier} (Complexity: {context.complexity}, Domain: {context.domain})"
        )

        # 2. Economy Mode Check
        try:
            remaining_budget_pct = self.budget_client.get_remaining_budget_percentage(user_id)
            if remaining_budget_pct < 0.10:
                logger.info(f"Economy Mode triggered for user {user_id} (Budget: {remaining_budget_pct:.2%})")
                if target_tier == ModelTier.TIER_2_SMART:
                    logger.info("Downgrading from Tier 2 to Tier 1 due to Economy Mode")
                    target_tier = ModelTier.TIER_1_FAST
        except Exception as e:
            # Fail Open: If budget check fails, proceed with baseline choice but log error
            logger.error(f"Failed to check budget for user {user_id}: {e}")

        # 3. Select Model from Registry
        candidates = self.registry.list_models(tier=target_tier)

        # Filter by static health check
        healthy_candidates = [m for m in candidates if m.is_healthy]

        # Filter by LoadBalancer (dynamic health check)
        if self.load_balancer:
            healthy_candidates = [m for m in healthy_candidates if self.load_balancer.is_provider_healthy(m.provider)]

        if not healthy_candidates:
            error_msg = f"No healthy models available for Tier: {target_tier}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Simple selection: Pick the first one.
        selected_model = healthy_candidates[0]
        logger.info(f"Routed to model: {selected_model.id} ({selected_model.provider})")

        return selected_model
