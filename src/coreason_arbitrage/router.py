# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_arbitrage

from typing import List, Optional

from coreason_identity.models import UserContext

from coreason_arbitrage.interfaces import BudgetClient
from coreason_arbitrage.load_balancer import LoadBalancer
from coreason_arbitrage.models import ModelDefinition, ModelTier, RoutingContext
from coreason_arbitrage.registry import ModelRegistry
from coreason_arbitrage.utils.logger import logger


class Router:
    """The Router determines the best model to use based on context, budget, and health.

    It executes the core logic of selecting the optimal model by evaluating
    complexity, domain specificity, economy mode, and provider health.
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

    def route(
        self,
        context: RoutingContext,
        user_context: Optional[UserContext] = None,
        excluded_providers: Optional[List[str]] = None,
    ) -> ModelDefinition:
        """Selects the optimal model for the given context and user.

        Logic:
        1. Determine Baseline Tier based on complexity and safety criticality.
        2. Verify Identity and Apply Economy Mode:
           - If user_context is missing, force Economy Mode (Tier 1).
           - If VIP (Executives), skip Economy Mode check.
           - Otherwise, check budget (downgrade Tier 2 to Tier 1 if budget < 10%).
        3. Check for Domain Priority matches (specialized models).
        4. Select Generic Model from Registry within the target Tier.

        All selections are filtered by:
        - `excluded_providers`: Explicit exclusions (e.g., failed providers in current request).
        - `LoadBalancer`: Dynamic health checks.

        Args:
            context: The routing context containing complexity and domain.
            user_context: The authenticated user context.
            excluded_providers: Optional list of provider names to exclude.

        Returns:
            ModelDefinition: The selected model configuration.

        Raises:
            RuntimeError: If no healthy models are available for the target requirements.
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

        # 2. Economy Mode / Identity Check
        if user_context is None:
            logger.warning("User context missing. Enforcing Economy Mode (Tier 1).")
            # Force downgrade to Tier 1 unless it's Tier 1 already.
            # Actually, constraint says "Force Economy Mode (Tier 1 models only)".
            # Does this mean I should downgrade even Tier 3?
            # "Tier selection, Economy Mode... Force Economy Mode (Tier 1 models only)".
            # Yes, "Tier 1 models only" implies stricter than standard Economy Mode (which downgrades T2->T1).
            # If I force Tier 1, I ensure safety.
            target_tier = ModelTier.TIER_1_FAST
        else:
            user_id = user_context.user_id

            # Check VIP Status
            # Case insensitive check for "Executives"
            is_vip = any(group.lower() == "executives" for group in user_context.groups)

            if is_vip:
                logger.info(f"VIP User detected ({user_id}). Skipping Economy Mode check.")
            else:
                try:
                    remaining_budget_pct = self.budget_client.get_remaining_budget_percentage(user_context)
                    if remaining_budget_pct < 0.10:
                        logger.info(f"Economy Mode triggered for user {user_id} (Budget: {remaining_budget_pct:.2%})")
                        if target_tier == ModelTier.TIER_2_SMART:
                            logger.info("Downgrading from Tier 2 to Tier 1 due to Economy Mode")
                            target_tier = ModelTier.TIER_1_FAST
                except Exception as e:
                    # Fail Open: If budget check fails, proceed with baseline choice but log error
                    logger.error(f"Failed to check budget for user {user_id}: {e}")

        # 3. Domain Priority Check
        if context.domain:
            domain_candidates = self.registry.list_models(domain=context.domain)
            # Filter by health
            healthy_domain_candidates = [m for m in domain_candidates if m.is_healthy]

            # Filter by excluded providers
            if excluded_providers:
                healthy_domain_candidates = [
                    m for m in healthy_domain_candidates if m.provider not in excluded_providers
                ]

            if self.load_balancer:
                healthy_domain_candidates = [
                    m for m in healthy_domain_candidates if self.load_balancer.is_provider_healthy(m.provider)
                ]

            if healthy_domain_candidates:
                # Attempt to find a match for the target tier
                tier_matches = [m for m in healthy_domain_candidates if m.tier == target_tier]

                selected_model: ModelDefinition
                if tier_matches:
                    selected_model = tier_matches[0]
                    logger.info(f"Domain match found (Tier Match). Routed to specialized model: {selected_model.id}")
                else:
                    # Fallback to any healthy domain model (Soft Fallback)
                    selected_model = healthy_domain_candidates[0]
                    logger.info(f"Domain match found (Tier Fallback). Routed to specialized model: {selected_model.id}")

                return selected_model

        # 4. Select Generic Model from Registry
        candidates = self.registry.list_models(tier=target_tier)

        # Filter by static health check
        healthy_candidates = [m for m in candidates if m.is_healthy]

        # Filter by excluded providers
        if excluded_providers:
            healthy_candidates = [m for m in healthy_candidates if m.provider not in excluded_providers]

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
