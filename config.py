"""Central configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# Provider: "anthropic" | "openai" | "ollama"
PROVIDER = os.getenv("PROVIDER", "anthropic")

# Two-tier model setup: AGENT for synthesis (quality), EVAL for formatting + eval (cost)
MODELS = {
    "anthropic": {
        "agent": "claude-sonnet-4-6",
        "eval":  "claude-haiku-4-5-20251001",
    },
    "openai": {
        "agent": "gpt-4.1",
        "eval":  "gpt-4.1-mini",
    },
    "ollama": {
        "agent": "llama3.1:70b",
        "eval":  "llama3.1:8b",
    },
}

if PROVIDER not in MODELS:
    raise ValueError(
        f"Unknown PROVIDER {PROVIDER!r}. Set PROVIDER to one of: "
        + ", ".join(MODELS)
    )

AGENT_MODEL = MODELS[PROVIDER]["agent"]
EVAL_MODEL  = MODELS[PROVIDER]["eval"]

CONFIDENCE_FLAG_THRESHOLD = 0.6
GROUNDING_PASS_THRESHOLD  = 0.7
CACHE_TTL_SECONDS = 3600

# Only the active provider's key is needed
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")
