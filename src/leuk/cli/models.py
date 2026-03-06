"""Model selection dialog for the leuk REPL."""

from __future__ import annotations

from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit.styles import Style

from leuk.providers.catalog import PROVIDER_NAMES

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


def run_model_selector(
    current_provider: str,
    current_model: str,
    provider_models: dict[str, list[tuple[str, str]]],
) -> tuple[str, str] | None:
    """Show a dialog to select a model.  Blocking.

    *provider_models* is a pre-fetched mapping of
    ``provider_key -> [(model_id, display_name), ...]``.

    Returns ``(provider_key, model_id)`` on selection, or ``None`` if
    the user cancels.
    """
    values: list[tuple[str, str]] = []
    default: str | None = None

    # Build a reverse lookup: model_id -> provider
    model_to_provider: dict[str, str] = {}

    # Determine display order: current provider first, then the rest
    providers_to_show: list[str] = []
    if current_provider in provider_models:
        providers_to_show.append(current_provider)
    for prov_key in provider_models:
        if prov_key != current_provider:
            providers_to_show.append(prov_key)

    if not providers_to_show:
        return None

    for prov_key in providers_to_show:
        provider_name = PROVIDER_NAMES.get(prov_key, prov_key)
        models = provider_models[prov_key]

        # Section header (disabled separator)
        values.append((f"__header__{prov_key}", f"--- {provider_name} ---"))

        for model_id, display_name in models:
            model_to_provider[model_id] = prov_key
            active = " *" if model_id == current_model else ""
            values.append((model_id, f"  {display_name}  ({model_id}){active}"))
            if model_id == current_model:
                default = model_id

    # If the current model isn't in any fetched catalog, add it at the top
    if default is None and current_model:
        values.insert(0, (current_model, f"  {current_model} (current) *"))
        default = current_model
        model_to_provider[current_model] = current_provider

    result = radiolist_dialog(
        title=HTML("<b>Select Model</b>"),
        text=HTML(
            "Use <b>arrow keys</b> to navigate, <b>Enter</b> to select, <b>Esc</b> to cancel."
        ),
        values=values,
        default=default,
        style=_DIALOG_STYLE,
    ).run()

    # Ignore header selections or cancellation
    if result is None or result.startswith("__header__"):
        return None

    prov = model_to_provider.get(result, current_provider)
    return (prov, result)
