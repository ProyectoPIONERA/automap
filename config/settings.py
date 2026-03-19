from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Simple role-based configuration for LLM usage across agents.
# You can override these via environment variables if needed.
ROLE_MODEL_ENV = {
    "schema_agent": "LLM_MODEL_SCHEMA",
    "mapper_agent": "LLM_MODEL_MAPPER",
    "yarrrml_architect": "LLM_MODEL_YARRRML",
    "refiner": "LLM_MODEL_REFINER",
}

ROLE_TEMPERATURE_ENV = {
    "schema_agent": "LLM_TEMP_SCHEMA",
    "mapper_agent": "LLM_TEMP_MAPPER",
    "yarrrml_architect": "LLM_TEMP_YARRRML",
    "refiner": "LLM_TEMP_REFINER",
}

# Sensible defaults if per-role env vars are not provided.
DEFAULT_MODEL = os.getenv("LLM_MODEL_DEFAULT", "qwen/qwen3-coder-30b")

DEFAULT_TEMPERATURES = {
    "schema_agent": 0.5,
    "mapper_agent": 0.4,
    "yarrrml_architect": 0.1,
    "refiner": 0.2,
}


def _resolve_model_for_role(role: str | None) -> str:
    """Return the model name for a given role, falling back to defaults.

    Priority:
    1. Role-specific env var (e.g. LLM_MODEL_YARRRML)
    2. COMET_MODEL (when LLM_PROVIDER is "comet")
    3. LLM_MODEL_DEFAULT
    4. Hard-coded DEFAULT_MODEL value
    """
    if role:
        model_env = ROLE_MODEL_ENV.get(role)
        if model_env:
            value = os.getenv(model_env)
            if value:
                return value
    # When using the Comet provider, COMET_MODEL overrides the default
    provider = os.getenv("LLM_PROVIDER", "lm_studio")
    if provider == "comet":
        comet_model = os.getenv("COMET_MODEL")
        if comet_model:
            return comet_model
    # Fallback to shared default
    return DEFAULT_MODEL


def _resolve_temperature_for_role(role: str | None) -> float:
    """Return the sampling temperature for a role, with a safe default."""
    if role:
        temp_env = ROLE_TEMPERATURE_ENV.get(role)
        if temp_env and os.getenv(temp_env) is not None:
            try:
                return float(os.getenv(temp_env))
            except ValueError:
                # If misconfigured, fall back to code defaults
                pass
        if role in DEFAULT_TEMPERATURES:
            return DEFAULT_TEMPERATURES[role]
    # Global fallback
    return 0.3


# ────────────────────────────────────────────────────────────────────
# Tiered retry-sampling configuration
# ────────────────────────────────────────────────────────────────────
# On the first attempt (retry_count == 0) the role's default temperature
# is used.  As retries increase the strategy progressively raises the
# temperature so the LLM doesn't keep producing the same broken output.
#
# Only universally-supported parameters (temperature, max_tokens) are
# used so the pipeline remains provider-agnostic (LM Studio, CometAPI,
# vLLM, OpenAI, etc.).
#
#   Tier 0 (retry 0)     – role defaults
#   Tier 1 (retry 1-2)   – slight temperature bump
#   Tier 2 (retry 3-4)   – moderate temperature bump
#   Tier 3 (retry 5+)    – maximum diversity
# ────────────────────────────────────────────────────────────────────

_DEFAULT_MAX_TOKENS = None          # let the model decide


def _resolve_retry_params(
    base_temperature: float,
    retry_count: int,
) -> dict:
    """Return a sampling-parameter dict for the given retry tier.

    Only uses ``temperature`` and ``max_tokens`` — parameters supported
    by every OpenAI-compatible provider.

    Returns
    -------
    dict with keys: temperature, max_tokens.
    """
    temperature = base_temperature
    max_tokens = _DEFAULT_MAX_TOKENS

    if retry_count == 0:
        # Tier 0 – role defaults
        pass

    elif retry_count <= 2:
        # Tier 1 – slight exploration
        temperature = min(base_temperature + 0.10, 0.55)
        max_tokens = 7000

    elif retry_count <= 4:
        # Tier 2 – moderate exploration
        temperature = min(base_temperature + 0.20, 0.65)
        max_tokens = 7000

    else:
        # Tier 3 – maximum diversity
        temperature = 0.8
        max_tokens = 7000

    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def get_llm(role: str | None = None, retry_count: int = 0):
    """Factory for creating a ChatOpenAI client.

    The ``role`` argument lets different agents use different models or
    base temperatures while sharing the same provider.

    The ``retry_count`` drives a **tiered sampling strategy** that
    progressively raises the temperature so the LLM doesn't keep
    producing the same broken output:

    * **Tier 0** (first attempt) – role defaults.
    * **Tier 1** (retry 1-2) – slight temperature bump.
    * **Tier 2** (retry 3-4) – moderate temperature bump.
    * **Tier 3** (retry 5+) – maximum diversity.

    Only universally-supported parameters (temperature, max_tokens) are
    used so the pipeline remains provider-agnostic.

    Roles currently used:
      - "schema_agent"
      - "mapper_agent"
      - "yarrrml_architect"
      - "refiner"
    """
    provider = os.getenv("LLM_PROVIDER", "lm_studio")

    model_name = _resolve_model_for_role(role)
    base_temperature = _resolve_temperature_for_role(role)

    # ── Resolve the sampling parameter set for this retry tier ──
    params = _resolve_retry_params(base_temperature, retry_count)

    temperature = params["temperature"]
    max_tokens = params["max_tokens"]

    # Log active retry parameters for observability
    if retry_count > 0:
        tier = (
            "Tier 1" if retry_count <= 2 else
            "Tier 2" if retry_count <= 4 else
            "Tier 3"
        )
        print(
            f"  [RETRY] [{role or 'default'}] retry #{retry_count} -> {tier}  "
            f"temp={temperature:.2f}"
            + (f"  max_tokens={max_tokens}" if max_tokens else "")
        )

    # ── Build the ChatOpenAI instance per provider ───────────────

    def _build_client(base_url: str, api_key: str, model: str) -> ChatOpenAI:
        """Shared constructor that wires sampling knobs."""
        kwargs: dict = dict(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    if provider == "lm_studio":
        lm_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
        base_url = lm_url.replace("/chat/completions", "")
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        return _build_client(base_url, "lm-studio", model_name)

    elif provider == "comet":
        return _build_client(
            "https://api.cometapi.com/v1",
            os.getenv("COMET_API_KEY", ""),
            model_name,
        )

    # Fallback: lm_studio-style configuration
    lm_url = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
    base_url = lm_url.replace("/chat/completions", "")
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    return _build_client(base_url, "lm-studio", model_name)


def get_llm_metadata() -> dict:
    """Return a snapshot of every agent's resolved model + temperature,
    plus the tiered retry-strategy parameters.

    Useful for stamping eval_metrics.json so you can trace which
    configuration produced which results.

    Returns
    -------
    dict – keys like ``provider``, ``default_model``,
           ``schema_agent_model``, ``schema_agent_temperature``,
           and ``retry_strategy_tiers`` with the full sampling config.
    """
    provider = os.getenv("LLM_PROVIDER", "lm_studio")
    roles = ["schema_agent", "mapper_agent", "yarrrml_architect", "refiner"]

    # Determine the effective default model (accounts for COMET_MODEL)
    effective_default = _resolve_model_for_role(None)

    meta: dict = {
        "llm_provider": provider,
        "llm_default_model": effective_default,
    }
    if provider == "comet" and os.getenv("COMET_MODEL"):
        meta["comet_model"] = os.getenv("COMET_MODEL")

    for role in roles:
        meta[f"{role}_model"] = _resolve_model_for_role(role)
        meta[f"{role}_temperature"] = _resolve_temperature_for_role(role)

    # Record the retry-strategy tiers so experiments are reproducible
    meta["retry_strategy_tiers"] = {
        "tier_0": "role defaults",
        "tier_1_retries": "1-2",
        "tier_1": _resolve_retry_params(0.2, 1),   # representative
        "tier_2_retries": "3-4",
        "tier_2": _resolve_retry_params(0.2, 3),
        "tier_3_retries": "5+",
        "tier_3": _resolve_retry_params(0.2, 5),
    }
    return meta

