"""Interactive authentication command for configuring API keys."""

from __future__ import annotations

import base64
import hashlib
import secrets
import webbrowser
from urllib.parse import urlencode

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from leuk.config import load_credentials, save_credentials

def _themed_console() -> Console:
    """A console bound to the **active** leuk theme (rebuilt to follow switches)."""
    from leuk.cli.theme import LEUK_THEME

    return Console(theme=LEUK_THEME)


# Bound to the active theme; refreshed on each run_auth() so /settings theme
# changes are reflected. Prompt/Confirm are passed this console explicitly.
console = _themed_console()

# Provider definitions: (key_name, display_name)
PROVIDERS = [
    ("zen", "OpenCode Zen"),
    ("anthropic", "Anthropic (Claude)"),
    ("openai", "OpenAI (GPT)"),
    ("google", "Google (Gemini)"),
    ("openrouter", "OpenRouter"),
    ("local", "Local (Ollama / llama.cpp / vLLM)"),
]

# ------------------------------------------------------------------
# Anthropic OAuth PKCE constants
# ------------------------------------------------------------------

_ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_ANTHROPIC_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
_ANTHROPIC_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_ANTHROPIC_REDIRECT_URI = "https://platform.claude.com/oauth/code/callback"
_ANTHROPIC_SCOPES = "user:profile user:inference"


# ------------------------------------------------------------------
# Abort sentinel — raised when user presses Ctrl+D or Ctrl+C
# ------------------------------------------------------------------


class _Abort(Exception):
    """Raised to signal the user wants to exit back to the REPL."""


def _ask(prompt: str, **kwargs: object) -> str:
    """Prompt.ask wrapper that converts EOFError/KeyboardInterrupt to _Abort."""
    try:
        return Prompt.ask(prompt, console=console, **kwargs)
    except (EOFError, KeyboardInterrupt):
        console.print()
        raise _Abort


def _confirm(prompt: str, **kwargs: object) -> bool:
    """Confirm.ask wrapper that converts EOFError/KeyboardInterrupt to _Abort."""
    try:
        return Confirm.ask(prompt, console=console, **kwargs)
    except (EOFError, KeyboardInterrupt):
        console.print()
        raise _Abort


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mask_key(key: str) -> str:
    """Mask an API key for display, showing only last 4 chars."""
    if len(key) <= 8:
        return "****"
    return f"...{key[-4:]}"


def _generate_pkce() -> tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge (S256).

    Returns (code_verifier, code_challenge).
    """
    # RFC 7636: 43-128 chars of [A-Z / a-z / 0-9 / - . _ ~]
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def _generate_state() -> str:
    """Generate a random state parameter for OAuth."""
    return secrets.token_urlsafe(64)


def _build_authorize_url(code_challenge: str, state: str) -> str:
    """Build the Anthropic OAuth authorization URL."""
    params = {
        "code": "true",
        "client_id": _ANTHROPIC_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _ANTHROPIC_REDIRECT_URI,
        "scope": _ANTHROPIC_SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{_ANTHROPIC_AUTHORIZE_URL}?{urlencode(params)}"


def _exchange_code_for_token(code: str, code_verifier: str, state: str) -> dict:
    """Exchange an authorization code for an access token.

    Sends a JSON body (not form-encoded) per the platform.claude.com token endpoint.
    Returns the full token response dict.
    """
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            _ANTHROPIC_TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": _ANTHROPIC_CLIENT_ID,
                "code": code,
                "redirect_uri": _ANTHROPIC_REDIRECT_URI,
                "code_verifier": code_verifier,
                "state": state,
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


def _refresh_access_token(refresh_token: str) -> dict:
    """Refresh an expired access token using a stored refresh token.

    Returns the full token response dict (contains new access_token and
    optionally a rotated refresh_token).
    """
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            _ANTHROPIC_TOKEN_URL,
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": _ANTHROPIC_CLIENT_ID,
                "scope": _ANTHROPIC_SCOPES,
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------------------
# Public refresh helper (called by providers on 401)
# ------------------------------------------------------------------


def refresh_anthropic_token() -> str | None:
    """Attempt to refresh the Anthropic OAuth credentials.

    1. Reads the stored refresh_token from credentials.
    2. Exchanges it for a new access_token via platform.claude.com.
    3. Persists the new access_token (and rotated refresh_token if any).

    Returns the new access_token on success, or None if no refresh_token
    is stored or the refresh fails.
    """
    creds = load_credentials()
    rt = creds.get("anthropic_refresh_token", "")
    if not rt:
        return None

    try:
        token_data = _refresh_access_token(rt)
    except httpx.HTTPError:
        return None

    new_access = token_data.get("access_token", "")
    if not new_access:
        return None

    creds["anthropic_auth_token"] = new_access

    # The server may rotate the refresh token — persist the new one if provided
    new_refresh = token_data.get("refresh_token", "")
    if new_refresh:
        creds["anthropic_refresh_token"] = new_refresh

    save_credentials(creds)
    return new_access


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def run_auth(current_provider: str = "anthropic") -> str | None:
    """Run the interactive authentication flow. Blocking (called from REPL via to_thread).

    Args:
        current_provider: The currently active provider key (e.g. "anthropic").

    Returns:
        The provider key to switch to, or None if unchanged / cancelled.
    """
    global console
    console = _themed_console()  # pick up the currently active theme
    try:
        return _run_auth_inner(current_provider)
    except _Abort:
        console.print("[dim]Cancelled.[/dim]")
        return None


def _has_credentials(creds: dict[str, str], key: str) -> bool:
    """Check whether a provider has stored credentials."""
    return bool(creds.get(f"{key}_api_key") or creds.get(f"{key}_auth_token"))


def _credential_summary(creds: dict[str, str], key: str) -> str:
    """Return a short status string for a provider's credentials."""
    if key == "local":
        # Local is configured by its endpoint, not a key (which is optional);
        # show the active base URL so a URL-only change is visible.
        from leuk.config import load_settings

        url = load_settings().llm.local_base_url
        api_key = creds.get("local_api_key", "")
        suffix = f" · key {_mask_key(api_key)}" if api_key else ""
        return f"[green]{url}{suffix}[/green]"
    api_key = creds.get(f"{key}_api_key", "")
    auth_token = creds.get(f"{key}_auth_token", "")
    if api_key:
        return f"[green]API key {_mask_key(api_key)}[/green]"
    if auth_token:
        return f"[green]auth token {_mask_key(auth_token)}[/green]"
    return "[dim]not configured[/dim]"


def _run_auth_inner(current_provider: str) -> str | None:
    """Provider list — pick a number to manage that provider's authorization.

    Selecting a provider by number is the single path to add / replace / delete
    its key (and switch to it). There are no separate a/e/d commands. Raises
    _Abort on Ctrl+D / Ctrl+C. Returns the provider key to switch to, or None.
    """
    while True:
        creds = load_credentials()
        console.print()
        console.print("[bold]Providers[/bold]  — pick a number to manage its authorization")
        console.print()
        for i, (key, name) in enumerate(PROVIDERS, 1):
            active = " [bold cyan]← active[/bold cyan]" if key == current_provider else ""
            console.print(f"  [bold]{i}[/bold]) {name}  {_credential_summary(creds, key)}{active}")
        console.print("  [bold]0[/bold]) Done")
        console.print()

        valid = [str(i) for i in range(len(PROVIDERS) + 1)]
        choice = _ask("Choice", choices=valid, default="0")
        if choice == "0":
            return None

        key, name = PROVIDERS[int(choice) - 1]
        switch_to = _manage_provider(key, name, current_provider)
        if switch_to:
            return switch_to
        # otherwise loop back to the provider list


def _configure_provider(creds: dict[str, str], key: str, name: str) -> None:
    """Run the provider-specific add/replace-credentials flow."""
    if key == "anthropic":
        _auth_anthropic(creds)
    elif key == "local":
        _auth_local(creds)
    else:
        _auth_generic(creds, key, name)


def _delete_provider(creds: dict[str, str], key: str, name: str, current_provider: str) -> None:
    """Delete a provider's stored credentials (with confirmation)."""
    if not _confirm(f"[red]Delete {name} credentials?[/red]", default=False):
        console.print("[dim]Cancelled.[/dim]")
        return
    for suffix in ("_api_key", "_auth_token", "_refresh_token", "_oauth_client_id"):
        creds.pop(f"{key}{suffix}", None)
    save_credentials(creds)
    console.print(f"[green]{name} credentials deleted.[/green]")
    if key == current_provider:
        console.print(
            "[yellow]Active provider credentials removed — switch to another "
            "provider or re-add them.[/yellow]"
        )


def _manage_provider(key: str, name: str, current_provider: str) -> str | None:
    """Manage one provider's auth. Returns its key if the user chose to switch."""
    # A fresh, unconfigured provider (other than local) → go straight to adding.
    if not _has_credentials(load_credentials(), key) and key != "local":
        creds = load_credentials()
        _configure_provider(creds, key, name)
        if _has_credentials(load_credentials(), key) and key != current_provider:
            if _confirm(f"Switch to {name} now?", default=True):
                return key
        return None

    # Configured (or local) → a small action menu.
    while True:
        creds = load_credentials()
        configured = _has_credentials(creds, key)
        console.print()
        console.print(f"[bold]{name}[/bold]  {_credential_summary(creds, key)}")
        console.print()

        opts: list[tuple[str, str]] = []
        if key == "local":
            opts.append(("c", "Set local endpoint (base URL / key)"))
        else:
            opts.append(("c", "Replace credentials"))
            if configured:
                opts.append(("d", "Delete credentials"))
        if key != current_provider:
            opts.append(("s", f"Switch to {name}"))
        opts.append(("0", "Back"))

        for k, label in opts:
            console.print(f"  [bold]{k}[/bold]) {label}")
        console.print()

        choice = _ask("Action", choices=[k for k, _ in opts], default="0")
        if choice == "0":
            return None
        if choice == "c":
            _configure_provider(creds, key, name)
        elif choice == "d":
            _delete_provider(creds, key, name, current_provider)
            if not _has_credentials(load_credentials(), key):
                return None  # nothing left to manage
        elif choice == "s":
            console.print(f"[green]Switching to {name}.[/green]")
            return key


# ------------------------------------------------------------------
# Anthropic auth
# ------------------------------------------------------------------


def _auth_anthropic(creds: dict[str, str]) -> None:
    """Anthropic-specific auth flow with multiple options."""
    console.print()
    console.print("[bold]Configure Anthropic (Claude)[/bold]")
    console.print()
    console.print("  [bold]1[/bold]) Claude Pro/Max subscription (OAuth login)")
    console.print("  [bold]2[/bold]) Create a new API key (instructions)")
    console.print("  [bold]3[/bold]) Enter API key manually")
    console.print("  [bold]0[/bold]) Cancel")
    console.print()

    choice = _ask("Method", choices=["0", "1", "2", "3"], default="3")

    if choice == "0":
        console.print("[dim]Cancelled.[/dim]")
        return

    if choice == "1":
        _auth_anthropic_oauth(creds)
        return

    if choice == "2":
        console.print()
        console.print(
            Panel(
                "[bold]To create an Anthropic API key:[/bold]\n\n"
                "1. Go to [link=https://console.anthropic.com/settings/keys]console.anthropic.com/settings/keys[/link]\n"
                "2. Click [bold]Create Key[/bold]\n"
                "3. Give it a name (e.g., 'leuk')\n"
                "4. Copy the key (starts with 'sk-ant-')\n"
                "5. Paste it below",
                title="[bold]Create API Key[/bold]",
                border_style="cyan",
                expand=False,
            )
        )
        console.print()

    # choice == "2" falls through to key entry, choice == "3" goes directly here
    api_key = _ask("Anthropic API key", password=False).strip()
    if not api_key:
        console.print("[dim]No key entered. Cancelled.[/dim]")
        return

    if not api_key.startswith("sk-ant-"):
        if not _confirm(
            "[yellow]Key doesn't start with 'sk-ant-'. Save anyway?[/yellow]",
            default=False,
        ):
            console.print("[dim]Cancelled.[/dim]")
            return

    creds["anthropic_api_key"] = api_key
    creds.pop("anthropic_auth_token", None)
    save_credentials(creds)
    console.print(f"[green]Saved Anthropic API key ({_mask_key(api_key)})[/green]")


def _auth_anthropic_oauth(creds: dict[str, str]) -> None:
    """Authenticate via Claude Pro/Max subscription using OAuth PKCE flow.

    1. Generate PKCE code_verifier + code_challenge
    2. Open the authorization URL in the user's browser
    3. User logs into claude.ai, authorizes, and gets a code
    4. User pastes the code here (may include #state suffix)
    5. Exchange the code for an access token
    6. Save the token
    """
    console.print()

    # Generate PKCE parameters
    code_verifier, code_challenge = _generate_pkce()
    state = _generate_state()
    auth_url = _build_authorize_url(code_challenge, state)

    # Try to open the URL in the default browser
    console.print("[dim]Opening browser...[/dim]")
    browser_opened = False
    try:
        browser_opened = webbrowser.open(auth_url)
    except Exception:
        pass

    if browser_opened:
        console.print(
            Panel(
                "[bold]Claude Pro/Max OAuth Login[/bold]\n\n"
                "A browser window has been opened.\n"
                "Log in with your Claude account and authorize access.\n\n"
                "If the browser didn't open, visit this URL manually:\n"
                f"[link={auth_url}]{auth_url}[/link]\n\n"
                "After authorizing, copy the code shown on the page\n"
                "and paste it below.",
                border_style="cyan",
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                "[bold]Claude Pro/Max OAuth Login[/bold]\n\n"
                "Open the following URL in your browser:\n\n"
                f"[link={auth_url}]{auth_url}[/link]\n\n"
                "Log in with your Claude account and authorize access.\n"
                "After authorizing, copy the code shown on the page\n"
                "and paste it below.",
                border_style="cyan",
                expand=False,
            )
        )
    console.print()

    raw_code = _ask("Authentication code").strip()
    if not raw_code:
        console.print("[dim]No code entered. Cancelled.[/dim]")
        return

    # The callback URL may include a #state fragment appended to the code.
    # Format: "<auth_code>#<returned_state>"
    if "#" in raw_code:
        auth_code, returned_state = raw_code.rsplit("#", 1)
        if returned_state != state:
            console.print("[red]State mismatch — possible CSRF. Aborting.[/red]")
            return
    else:
        auth_code = raw_code

    # Exchange code for token
    console.print("[dim]Exchanging code for access token...[/dim]")
    try:
        token_data = _exchange_code_for_token(auth_code, code_verifier, state)
    except httpx.HTTPStatusError as exc:
        console.print(f"[red]Token exchange failed: {exc.response.status_code}[/red]")
        try:
            body = exc.response.json()
            console.print(f"[red]{body.get('error_description', body.get('error', ''))}[/red]")
        except Exception:
            console.print(f"[red]{exc.response.text[:200]}[/red]")
        return
    except httpx.HTTPError as exc:
        console.print(f"[red]Token exchange failed: {exc}[/red]")
        return

    access_token = token_data.get("access_token", "")
    if not access_token:
        console.print("[red]No access_token in response.[/red]")
        return

    # Store tokens
    creds["anthropic_auth_token"] = access_token
    refresh_token = token_data.get("refresh_token", "")
    if refresh_token:
        creds["anthropic_refresh_token"] = refresh_token

    # Remove stale keys from older auth flows
    creds.pop("anthropic_api_key", None)
    creds.pop("anthropic_oauth_client_id", None)

    save_credentials(creds)
    console.print(f"[green]Authenticated! OAuth token saved ({_mask_key(access_token)})[/green]")
    console.print("[dim]leuk will use your Claude Pro/Max subscription for API calls.[/dim]")


# ------------------------------------------------------------------
# Generic / Local auth
# ------------------------------------------------------------------


def _auth_generic(creds: dict[str, str], provider_key: str, provider_name: str) -> None:
    """Generic API key entry for OpenAI, Google, OpenRouter."""
    console.print()
    console.print(f"[bold]Configure {provider_name}[/bold]")

    current = creds.get(f"{provider_key}_api_key", "")
    if current:
        console.print(f"[dim]Current key: {_mask_key(current)}[/dim]")

    console.print()
    api_key = _ask(f"{provider_name} API key", password=False).strip()
    if not api_key:
        console.print("[dim]No key entered. Cancelled.[/dim]")
        return

    creds[f"{provider_key}_api_key"] = api_key
    save_credentials(creds)
    console.print(f"[green]Saved {provider_name} API key ({_mask_key(api_key)})[/green]")


def _auth_local(creds: dict[str, str]) -> None:
    """Configure the local endpoint: base URL (config.json) + optional API key.

    The base URL is configuration, not a credential — it persists to
    ``config.json`` (``llm.local_base_url``), which the REPL reloads right
    after ``/auth`` so the change applies immediately.
    """
    from leuk.config import load_settings, save_persistent_config

    console.print()
    console.print("[bold]Configure Local Model (Ollama / llama.cpp / vLLM)[/bold]")
    console.print(
        "[dim]Any OpenAI-compatible endpoint works — e.g. Ollama "
        "(http://localhost:11434/v1) or llama-server (http://localhost:8080/v1). "
        "Ollama and llama.cpp usually need no API key; vLLM may.[/dim]"
    )
    console.print()

    # 1. Base URL → config.json (llm.local_base_url).
    current_url = load_settings().llm.local_base_url
    base_url = _ask(
        f"Base URL [{current_url}]",
        default=current_url,
        password=False,
    ).strip()
    if base_url and base_url != current_url:
        save_persistent_config({"llm": {"local_base_url": base_url}})
        # The model list is cached per provider; a new endpoint serves a
        # different catalog, so drop the stale "local" cache.
        from leuk.providers.catalog import invalidate_cache

        invalidate_cache("local")
        console.print(f"[green]Saved base URL: {base_url}[/green]")
    else:
        console.print(f"[dim]Base URL unchanged ({current_url}).[/dim]")

    # 2. API key → credentials.json.
    current_key = creds.get("local_api_key", "")
    if current_key:
        console.print(f"[dim]Current key: {_mask_key(current_key)} — enter '-' to clear.[/dim]")
    api_key = _ask(
        "API key (empty = keep current)" if current_key else "API key (empty = none)",
        default="",
        password=False,
    ).strip()

    if api_key == "-" and current_key:
        creds.pop("local_api_key", None)
        save_credentials(creds)
        console.print("[green]Cleared the local API key.[/green]")
    elif api_key:
        creds["local_api_key"] = api_key
        save_credentials(creds)
        console.print(f"[green]Saved local API key ({_mask_key(api_key)})[/green]")
    else:
        console.print("[dim]API key unchanged.[/dim]")
