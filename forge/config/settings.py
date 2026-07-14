"""
FORGE Settings — loads config.yaml, validates, exposes typed config.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ── Config paths ──────────────────────────────────────────────────────────────
FORGE_HOME      = Path.home() / ".forge"
CONFIG_PATH     = FORGE_HOME / "config.yaml"
LOCAL_CONFIG    = Path(".forge") / "config.yaml"   # project-level override
MEMORY_DB       = FORGE_HOME / "memory.db"
SKILLS_DIR      = FORGE_HOME / "skills"
LOG_FILE        = FORGE_HOME / "forge.log"


# ── Pydantic models ───────────────────────────────────────────────────────────
class ProviderConfig(BaseModel):
    api_key:    str   = ""
    enabled:    bool  = True
    tier:       str   = "smart_free"
    models:     dict  = Field(default_factory=dict)
    base_url:   str   = ""
    rpm_limit:  int   = 20
    tpd_limit:  int   = 0        # 0 = unlimited


class CompressorConfig(BaseModel):
    enabled:             bool = True
    shell_output:        bool = True
    signature_mode:      bool = True
    ast_compression:     bool = True
    simhash_dedup:       bool = True
    max_context_tokens:  int  = 4000
    signature_cache_ttl: int  = 3600


class MemoryConfig(BaseModel):
    enabled:             bool = True
    backend:             str  = "sqlite"
    path:                str  = str(MEMORY_DB)
    skill_auto_generate: bool = True
    skill_threshold:     int  = 5
    session_persist:     bool = True


class OutputConfig(BaseModel):
    verbosity_trim:  bool = True
    stream:          bool = True
    show_token_stats:bool = True
    show_provider:   bool = True
    show_cost_saved: bool = True


class ForgeConfig(BaseModel):
    providers:  dict[str, ProviderConfig] = Field(default_factory=dict)
    routing:    dict[str, Any]            = Field(default_factory=dict)
    compressor: CompressorConfig          = Field(default_factory=CompressorConfig)
    memory:     MemoryConfig              = Field(default_factory=MemoryConfig)
    output:     OutputConfig              = Field(default_factory=OutputConfig)


# ── Loader ────────────────────────────────────────────────────────────────────
def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config() -> ForgeConfig:
    """
    Priority: local .forge/config.yaml > ~/.forge/config.yaml > env vars
    """
    base   = _load_yaml(CONFIG_PATH)
    local  = _load_yaml(LOCAL_CONFIG)

    # Deep merge: local overrides base
    merged = _deep_merge(base, local)

    # Env var overrides for API keys (FORGE_GROQ_KEY, FORGE_GEMINI_KEY, etc.)
    providers_raw = merged.get("providers", {})
    for name, cfg in providers_raw.items():
        env_key = f"FORGE_{name.upper()}_KEY"
        if val := os.getenv(env_key):
            cfg["api_key"] = val

    # Build typed config
    providers = {
        name: ProviderConfig(**cfg)
        for name, cfg in providers_raw.items()
    }

    return ForgeConfig(
        providers=providers,
        routing=merged.get("routing", {}),
        compressor=CompressorConfig(**merged.get("compressor", {})),
        memory=MemoryConfig(**merged.get("memory", {})),
        output=OutputConfig(**merged.get("output", {})),
    )


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def ensure_forge_home() -> None:
    """Create ~/.forge directory structure if missing."""
    FORGE_HOME.mkdir(exist_ok=True)
    SKILLS_DIR.mkdir(exist_ok=True)
    LOG_FILE.parent.mkdir(exist_ok=True)
