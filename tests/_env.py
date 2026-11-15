"""The provider environment the suite runs with.

No test reaches a real model on its own. Whatever the shell exports, tests
see what CI gives them: OpenAI as the provider with no key, and no other
provider setting. A local Ollama needs no key, so it is pointed at a port
nothing listens on. An app a test builds finds its provider unavailable,
which every path that calls one already handles.

BANKO_LIVE_MODEL_TESTS=1 leaves the shell as it is, for the few script-style
files that exist to call a real model."""

from collections.abc import MutableMapping

OPT_IN = "BANKO_LIVE_MODEL_TESTS"
NOWHERE = "http://127.0.0.1:9"  # the discard port: a connection is refused at once


def provider_variables() -> set[str]:
    """Every environment name a provider setting is read from, old names included."""
    from banko_ai.config.settings import Settings, field_meta
    names: set[str] = set()
    for field in Settings.model_fields.values():
        meta = field_meta(field)
        if meta.group == "ai":
            names.update((meta.env, *meta.aliases))
    return names


def without_providers(environ: MutableMapping[str, str]) -> None:
    if environ.get(OPT_IN) == "1":
        return
    # BANKO_CONFIG names a YAML file that can carry any setting, credentials included.
    unwanted = provider_variables() | {"BANKO_CONFIG"}
    for name in [name for name in environ if name.upper() in unwanted]:  # settings match names in any case
        del environ[name]
    environ["AI_SERVICE"] = "openai"
    environ["OLLAMA_HOST"] = NOWHERE
