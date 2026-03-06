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

console = Console()

# Provider definitions: (key_name, display_name)
PROVIDERS = [
    ("zen", "OpenCode Zen"),
    ("anthropic", "Anthropic (Claude)"),
    ("openai", "OpenAI (GPT)"),
    ("google", "Google (Gemini)"),
    ("openrouter", "OpenRouter"),
    ("local", "Local (vLLM / Ollama)"),
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
        return Prompt.ask(prompt, **kwargs)
    except (EOFError, KeyboardInterrupt):
        console.print()
        raise _Abort


def _confirm(prompt: str, **kwargs: object) -> bool:
    """Confirm.ask wrapper that converts EOFError/KeyboardInterrupt to _Abort."""
    try:
        return Confirm.ask(prompt, **kwargs)
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
    api_key = creds.get(f"{key}_api_key", "")
    auth_token = creds.get(f"{key}_auth_token", "")
    if api_key:
        return f"[green]API key {_mask_key(api_key)}[/green]"
    if auth_token:
        return f"[green]auth token {_mask_key(auth_token)}[/green]"
    return "[dim]not configured[/dim]"


def _run_auth_inner(current_provider: str) -> str | None:
    """Inner auth flow — raises _Abort on Ctrl+D / Ctrl+C.

    Returns the provider key to switch to, or None if unchanged.
    """
    creds = load_credentials()

    console.print()
    console.print("[bold]Select a provider[/bold]  (number to switch, or action below)")
    console.print()

    for i, (key, name) in enumerate(PROVIDERS, 1):
        active = " [bold cyan]*[/bold cyan]" if key == current_provider else ""
        status = _credential_summary(creds, key)
        console.print(f"  [bold]{i}[/bold]) {name}  {status}{active}")

    console.print()
    console.print(
        "  [bold]a[/bold])dd  [bold]e[/bold])dit  [bold]d[/bold])elete  [bold]0[/bold]) cancel"
    )
    console.print()

    valid = [str(i) for i in range(len(PROVIDERS) + 1)] + ["a", "e", "d"]
    choice = _ask("Choice", choices=valid, default="0")

    if choice == "0":
        return None

    # ---- Switch provider ------------------------------------------------
    if choice.isdigit() and int(choice) >= 1:
        idx = int(choice) - 1
        key, name = PROVIDERS[idx]
        if not _has_credentials(creds, key) and key != "local":
            console.print(
                f"[yellow]{name} has no credentials. Configure it first with (a)dd.[/yellow]"
            )
            return None
        if key == current_provider:
            console.print(f"[dim]{name} is already active.[/dim]")
            return None
        console.print(f"[green]Switched to {name}.[/green]")
        return key

    # ---- Add / Edit credentials -----------------------------------------
    if choice in ("a", "e"):
        return _run_configure(creds, current_provider)

    # ---- Delete credentials ---------------------------------------------
    if choice == "d":
        return _run_delete(creds, current_provider)

    return None


def _run_configure(creds: dict[str, str], current_provider: str) -> str | None:
    """Sub-menu: add or edit credentials for a provider."""
    console.print()
    console.print("[bold]Configure which provider?[/bold]")
    for i, (key, name) in enumerate(PROVIDERS, 1):
        status = _credential_summary(creds, key)
        console.print(f"  [bold]{i}[/bold]) {name}  {status}")
    console.print("  [bold]0[/bold]) Cancel")
    console.print()

    choice = _ask("Provider", choices=[str(i) for i in range(len(PROVIDERS) + 1)], default="0")
    if choice == "0":
        return None

    idx = int(choice) - 1
    provider_key, provider_name = PROVIDERS[idx]

    if provider_key == "anthropic":
        _auth_anthropic(creds)
    elif provider_key == "local":
        _auth_local(creds)
    else:
        _auth_generic(creds, provider_key, provider_name)

    # If the configured provider now has credentials and is different from
    # the current one, offer to switch.
    if provider_key != current_provider and _has_credentials(creds, provider_key):
        if _confirm(f"Switch to {provider_name} now?", default=True):
            return provider_key

    return None


def _run_delete(creds: dict[str, str], current_provider: str) -> str | None:
    """Sub-menu: delete stored credentials for a provider."""
    # Only show providers that have credentials
    configured = [
        (i, key, name) for i, (key, name) in enumerate(PROVIDERS, 1) if _has_credentials(creds, key)
    ]
    if not configured:
        console.print("[dim]No credentials to delete.[/dim]")
        return None

    console.print()
    console.print("[bold]Delete credentials for which provider?[/bold]")
    for i, key, name in configured:
        console.print(f"  [bold]{i}[/bold]) {name}  {_credential_summary(creds, key)}")
    console.print("  [bold]0[/bold]) Cancel")
    console.print()

    valid = [str(i) for i, _, _ in configured] + ["0"]
    choice = _ask("Provider", choices=valid, default="0")
    if choice == "0":
        return None

    idx = int(choice) - 1
    key, name = PROVIDERS[idx]

    if not _confirm(f"[red]Delete {name} credentials?[/red]", default=False):
        console.print("[dim]Cancelled.[/dim]")
        return None

    # Remove all credential keys for this provider
    for suffix in ("_api_key", "_auth_token", "_refresh_token", "_oauth_client_id"):
        creds.pop(f"{key}{suffix}", None)
    save_credentials(creds)
    console.print(f"[green]{name} credentials deleted.[/green]")

    # If we deleted the active provider's credentials, warn the user
    if key == current_provider:
        console.print(
            "[yellow]Active provider credentials removed. "
            "Switch to another provider or add new credentials.[/yellow]"
        )

    return None


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
    api_key = _ask("Anthropic API key", password=True).strip()
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
    api_key = _ask(f"{provider_name} API key", password=True).strip()
    if not api_key:
        console.print("[dim]No key entered. Cancelled.[/dim]")
        return

    creds[f"{provider_key}_api_key"] = api_key
    save_credentials(creds)
    console.print(f"[green]Saved {provider_name} API key ({_mask_key(api_key)})[/green]")


def _auth_local(creds: dict[str, str]) -> None:
    """Configure local model (vLLM / Ollama) settings."""
    console.print()
    console.print("[bold]Configure Local Model (vLLM / Ollama)[/bold]")
    console.print("[dim]Ollama usually needs no API key. vLLM may require one.[/dim]")
    console.print()

    current = creds.get("local_api_key", "")
    if current:
        console.print(f"[dim]Current key: {_mask_key(current)}[/dim]")

    api_key = _ask(
        "API key (leave empty to skip)",
        default="",
        password=True,
    ).strip()

    if api_key:
        creds["local_api_key"] = api_key
        save_credentials(creds)
        console.print(f"[green]Saved local API key ({_mask_key(api_key)})[/green]")
    else:
        console.print("[dim]Skipped.[/dim]")
