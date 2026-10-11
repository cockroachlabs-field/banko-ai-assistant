"""The provider persona.

Banko can run on one provider (Ollama, offline) while every surface a user
sees shows another (watsonx). Everything that shows provider identity reads
it from here, so the persona is consistent. The real provider stays in the
boot log and in /api/health under "actual", so an engineer can always check
what is serving requests.
"""

from __future__ import annotations

from dataclasses import dataclass

PROVIDER_BRANDS: dict[str, tuple[str, str]] = {
    "watsonx": ("IBM watsonx", "watsonx-icon.svg"),
    "openai": ("OpenAI", "openai-icon.svg"),
    "aws": ("AWS Bedrock", "aws-bedrock-icon.svg"),
    "gemini": ("Google Gemini", "google-gemini-icon.svg"),
    "ollama": ("Ollama", "local-model-icon.svg"),
}

NEUTRAL_UNAVAILABLE = "The assistant couldn't reach its AI model just now. Please try again in a moment."
NEUTRAL_FAILED = "Something went wrong on our side. Please try again."


@dataclass(frozen=True)
class Identity:
    provider: str
    name: str
    icon_file: str
    model: str


def _brand(provider: str) -> tuple[str, str]:
    return PROVIDER_BRANDS.get(provider, (provider, "local-model-icon.svg"))


def actual_identity(service: str, model: str) -> Identity:
    name, icon = _brand(service)
    return Identity(provider=service, name=name, icon_file=icon, model=model or "unknown")


def _listed(cfg) -> list[str]:
    return [m.strip() for m in (cfg.ai_display_models or "").split(",") if m.strip()]


def display_models(cfg, actual_models: list[str] | None) -> list[str]:
    if not cfg.ai_display_provider:
        return list(actual_models or [])
    listed = _listed(cfg)
    if listed:
        return listed
    if cfg.ai_display_model:
        return [cfg.ai_display_model]
    return list(actual_models or [])


def display_identity(cfg, actual: Identity, *, selected_model: str | None = None) -> Identity:
    if not cfg.ai_display_provider:
        return actual
    name, icon = _brand(cfg.ai_display_provider)
    model = selected_model or cfg.ai_display_model or (_listed(cfg) or [actual.model])[0]
    return Identity(provider=cfg.ai_display_provider, name=name, icon_file=icon, model=model)


def display_label(cfg, value: str | None) -> str | None:
    """Map a stored provider_used value (coach nudges) to what the UI shows."""
    if not cfg.ai_display_provider or not value or value == "fallback":
        return value
    return cfg.ai_display_provider


def ui_context(identity: Identity, *, status: str) -> dict[str, str]:
    return {
        "name": identity.name,
        "current_service": identity.provider.upper(),
        "current_model": identity.model or "Unknown",
        "status": status,
        "icon_file": identity.icon_file,
        "icon_alt": identity.name,
        "icon": "🧠",
    }


def neutral_error(exc: BaseException | str | None = None) -> str:
    """User-facing copy for any provider failure. It never echoes the
    exception, which can name the real provider or its host; the details go
    to the server log at the call site."""
    return NEUTRAL_UNAVAILABLE
