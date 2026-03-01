"""Model selection dialog for the leuk REPL."""

from __future__ import annotations

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.styles import Style

# ------------------------------------------------------------------
# Model catalog: provider -> [(model_id, display_name), ...]
#
# Only the most useful models are listed per provider.  The user can
# always set an arbitrary model via LEUK_LLM_MODEL env var or
# config.env.
# ------------------------------------------------------------------

PROVIDER_MODELS: dict[str, list[tuple[str, str]]] = {
    "anthropic": [
        ("claude-sonnet-4-20250514", "Claude Sonnet 4"),
        ("claude-opus-4-20250514", "Claude Opus 4"),
        ("claude-3-7-sonnet-20250219", "Claude 3.7 Sonnet"),
        ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
    ],
    "openai": [
        ("gpt-4.1", "GPT-4.1"),
        ("gpt-4.1-mini", "GPT-4.1 Mini"),
        ("gpt-4.1-nano", "GPT-4.1 Nano"),
        ("o3", "o3"),
        ("o4-mini", "o4 Mini"),
    ],
    "google": [
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("gemini-2.0-flash", "Gemini 2.0 Flash"),
    ],
    "openrouter": [
        ("anthropic/claude-sonnet-4", "Claude Sonnet 4"),
        ("anthropic/claude-opus-4", "Claude Opus 4"),
        ("openai/gpt-4.1", "GPT-4.1"),
        ("google/gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("deepseek/deepseek-r1", "DeepSeek R1"),
        ("deepseek/deepseek-chat-v3-0324", "DeepSeek V3"),
    ],
    "local": [
        ("llama3.1:8b", "Llama 3.1 8B"),
        ("llama3.1:70b", "Llama 3.1 70B"),
        ("qwen2.5:7b", "Qwen 2.5 7B"),
        ("qwen2.5:32b", "Qwen 2.5 32B"),
        ("deepseek-r1:8b", "DeepSeek R1 8B"),
        ("mistral:7b", "Mistral 7B"),
    ],
}

# Display names for providers
_PROVIDER_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "openrouter": "OpenRouter",
    "local": "Local",
}

_DIALOG_STYLE = Style.from_dict(
    {
        "dialog": "bg:#1a1a2e",
        "dialog frame.label": "bg:#16213e #e0e0e0 bold",
        "dialog.body": "bg:#1a1a2e #e0e0e0",
        "dialog shadow": "bg:#0f0f0f",
        "button": "bg:#16213e #e0e0e0",
        "button.focused": "bg:#0f3460 #ffffff bold",
        "radio-list": "bg:#1a1a2e #e0e0e0",
        "radio": "#00aa00",
        "radio-checked": "#00ff00 bold",
    }
)


def get_available_models(
    provider: str,
    creds: dict[str, str],
) -> list[tuple[str, str]]:
    """Return models available for the given provider.

    Only returns results for providers that have configured credentials
    (or 'local' which needs none).
    """
    has_creds = bool(
        creds.get(f"{provider}_api_key")
        or creds.get(f"{provider}_auth_token")
        or provider == "local"
    )
    if not has_creds:
        return []
    return PROVIDER_MODELS.get(provider, [])


def run_model_selector(
    current_provider: str,
    current_model: str,
    creds: dict[str, str],
) -> str | None:
    """Show a dialog to select a model.  Blocking.

    Builds a single list of models grouped by provider, but only includes
    providers that have credentials configured.

    Returns the selected model ID, or None if cancelled.
    """
    values: list[tuple[str, str]] = []
    default: str | None = None

    # Determine which providers to show
    providers_to_show: list[str] = []

    # Current provider first
    if current_provider in PROVIDER_MODELS:
        providers_to_show.append(current_provider)

    # Then other configured providers
    for prov_key in PROVIDER_MODELS:
        if prov_key == current_provider:
            continue
        has_creds = bool(
            creds.get(f"{prov_key}_api_key")
            or creds.get(f"{prov_key}_auth_token")
            or prov_key == "local"
        )
        if has_creds:
            providers_to_show.append(prov_key)

    if not providers_to_show:
        return None

    for prov_key in providers_to_show:
        provider_name = _PROVIDER_NAMES.get(prov_key, prov_key)
        models = PROVIDER_MODELS[prov_key]

        # Add a section header as a disabled separator
        values.append((f"__header__{prov_key}", f"--- {provider_name} ---"))

        for model_id, display_name in models:
            active = " *" if model_id == current_model else ""
            values.append((model_id, f"  {display_name}  ({model_id}){active}"))
            if model_id == current_model:
                default = model_id

    # If the current model isn't in any catalog, add it at the top
    if default is None and current_model:
        provider_name = _PROVIDER_NAMES.get(current_provider, current_provider)
        values.insert(0, (current_model, f"  {current_model} (current) *"))
        default = current_model

    result = radiolist_dialog(
        title=HTML("<b>Select Model</b>"),
        text=HTML("Use <b>arrow keys</b> to navigate, <b>Enter</b> to select, <b>Esc</b> to cancel."),
        values=values,
        default=default,
        style=_DIALOG_STYLE,
    ).run()

    # Ignore header selections
    if result and result.startswith("__header__"):
        return None

    return result
