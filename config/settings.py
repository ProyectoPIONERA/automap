from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Simple role-based configuration for LLM usage across agents.
# You can override these via environment variables if needed.
ROLE_MODEL_ENV = {
    "schema_agent": "LLM_MODEL_SCHEMA",
    "mapper_agent": "LLM_MODEL_MAPPER",
    "schema_alignment": "LLM_MODEL_ALIGNMENT",
    "yarrrml_architect": "LLM_MODEL_YARRRML",
    "prefix_agent": "LLM_MODEL_YARRRML",
    "entity_builder": "LLM_MODEL_YARRRML",
    "relationship_linker": "LLM_MODEL_YARRRML",
    "cq_validator": "LLM_MODEL_CQ",
    "refiner": "LLM_MODEL_REFINER",
}

ROLE_TEMPERATURE_ENV = {
    "schema_agent": "LLM_TEMP_SCHEMA",
    "mapper_agent": "LLM_TEMP_MAPPER",
    "schema_alignment": "LLM_TEMP_ALIGNMENT",
    "yarrrml_architect": "LLM_TEMP_YARRRML",
    "prefix_agent": "LLM_TEMP_YARRRML",
    "entity_builder": "LLM_TEMP_YARRRML",
    "relationship_linker": "LLM_TEMP_YARRRML",
    "cq_validator": "LLM_TEMP_CQ",
    "refiner": "LLM_TEMP_REFINER",
}

# Sensible defaults if per-role env vars are not provided.
DEFAULT_MODEL = os.getenv("LLM_MODEL_DEFAULT", "qwen2.5-coder-14b-instruct")

DEFAULT_TEMPERATURES = {
    "schema_agent": 0.3,
    "mapper_agent": 0.3,
    "schema_alignment": 0.2,
    "yarrrml_architect": 0.3,
    "prefix_agent": 0.2,
    "entity_builder": 0.3,
    "relationship_linker": 0.3,
    "cq_validator": 0.2,
    "refiner": 0.2,
}

# Per-role timeout defaults (seconds).  Capped at 280s to stay under
# LM Studio's hard 300s server-side timeout.  Override globally with
# the LLM_TIMEOUT env var.
DEFAULT_TIMEOUTS = {
    "schema_agent": 120,
    "mapper_agent": 120,
    "schema_alignment": 280,
    "yarrrml_architect": 280,
    "prefix_agent": 120,
    "entity_builder": 280,
    "relationship_linker": 280,
    "cq_validator": 280,
    "refiner": 280,
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


def get_llm(role: str | None = None):
    """Factory for creating a ChatOpenAI client.

    The ``role`` argument lets different agents use different models or
    temperatures while sharing the same provider.

    Fixed temperatures are used per role — no dynamic retry-based
    temperature changes.  Low temperatures ensure local LLMs follow
    the one-shot YARRRML examples precisely.

    Roles currently used:
      - "schema_agent"
      - "mapper_agent"
      - "prefix_agent"
      - "entity_builder"
      - "relationship_linker"
      - "refiner"
    """
    provider = os.getenv("LLM_PROVIDER", "lm_studio")

    model_name = _resolve_model_for_role(role)
    temperature = _resolve_temperature_for_role(role)

    # Per-role timeout: env override → role default → global fallback
    global_timeout = float(os.getenv("LLM_TIMEOUT", "0"))
    role_timeout = DEFAULT_TIMEOUTS.get(role, 300) if role else 300
    timeout = global_timeout if global_timeout > 0 else role_timeout

    # ── Build the ChatOpenAI instance per provider ───────────────

    def _build_client(base_url: str, api_key: str, model: str) -> ChatOpenAI:
        """Shared constructor."""
        return ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )

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


def get_llm_with_retry(role: str | None = None, max_retries: int = 2):
    """Return an LLM client wrapped with retry logic.

    Uses LangChain's ``with_retry`` to automatically retry on transient
    failures (timeouts, connection errors, server errors) with exponential
    backoff.  Falls back to the standard ``get_llm()`` client.

    Parameters
    ----------
    role : str | None
        Agent role (same as ``get_llm``).
    max_retries : int
        Maximum number of retry attempts (default 2 — total 3 calls).
    """
    import openai

    llm = get_llm(role=role)
    return llm.with_retry(
        retry_if_exception_type=(
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ),
        stop_after_attempt=max_retries,
        wait_exponential_jitter=True,
    )


def get_llm_metadata() -> dict:
    """Return a snapshot of every agent's resolved model + temperature.

    Useful for stamping eval_metrics.json so you can trace which
    configuration produced which results.

    Returns
    -------
    dict – keys like ``provider``, ``default_model``,
           ``schema_agent_model``, ``schema_agent_temperature``, etc.
    """
    provider = os.getenv("LLM_PROVIDER", "lm_studio")
    roles = [
        "schema_agent", "mapper_agent", "schema_alignment",
        "prefix_agent", "entity_builder", "relationship_linker",
        "cq_validator", "refiner",
    ]

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

    return meta
