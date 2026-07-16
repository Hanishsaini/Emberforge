"""
EMBERFORGE — Free, Open-source, Routing & Generation Engine
A self-improving agentic coding harness. Built to replace Claude Code.

github.com/Hanishsaini/emberforge
"""

__version__ = "0.1.0"
__author__  = "Honey Stark"
__license__ = "MIT"

TIER_LOCAL      = "local"
TIER_FAST_FREE  = "fast_free"
TIER_SMART_FREE = "smart_free"
TIER_BEST_FREE  = "best_free"
TIER_PAID       = "paid"

TIER_ORDER = [TIER_LOCAL, TIER_FAST_FREE, TIER_SMART_FREE, TIER_BEST_FREE, TIER_PAID]

TASK_AUTOCOMPLETE  = "autocomplete"
TASK_EXPLAIN       = "explain"
TASK_DEBUG         = "debug"
TASK_TEST          = "test"
TASK_REFACTOR      = "refactor"
TASK_WRITE         = "write"
TASK_ARCHITECTURE  = "architecture"
TASK_RESEARCH      = "research"
TASK_REVIEW        = "review"