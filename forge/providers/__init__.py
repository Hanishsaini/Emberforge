"""FORGE Provider Registry — builds all providers from config."""
from __future__ import annotations

from forge.config.settings import ForgeConfig, ProviderConfig
from forge.providers.base import BaseProvider
from forge.providers.openai_compat import OpenAICompatProvider
from forge import TIER_ORDER


def build_providers(config: ForgeConfig) -> dict[str, BaseProvider]:
    providers: dict[str, BaseProvider] = {}
    for name, cfg in config.providers.items():
        if not cfg.enabled:
            continue
        if not cfg.api_key and name != "ollama":
            continue
        providers[name] = OpenAICompatProvider(
            name=name,
            api_key=cfg.api_key or "ollama",
            tier=cfg.tier,
            models=cfg.models,
            base_url=cfg.base_url,
            rpm_limit=cfg.rpm_limit,
        )
    return providers


def get_providers_by_tier(
    providers: dict[str, BaseProvider], tier: str
) -> list[BaseProvider]:
    tier_providers = [p for p in providers.values() if p.tier == tier and p.is_available()]
    return sorted(tier_providers, key=lambda p: p.health.avg_latency_ms)


def get_providers_at_or_above_tier(
    providers: dict[str, BaseProvider], min_tier: str
) -> list[BaseProvider]:
    result = []
    start = TIER_ORDER.index(min_tier)
    for tier in TIER_ORDER[start:]:
        result.extend(get_providers_by_tier(providers, tier))
    return result
