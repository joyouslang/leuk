#!/usr/bin/env bash
#
# setup-input-control.sh — one-shot setup for leuk's desktop-control tool.
#
# Installs/builds ydotool >= 1.0 (+ ydotoold), a screenshot backend, grants
# /dev/uinput access, and runs ydotoold as a user service. Re-runnable.
#
# Usage:   bash scripts/setup-input-control.sh
# Verify:  leuk doctor
#
set -euo pipefail

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!  \033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m✓  \033[0m %s\n' "$*"; }

if [[ "$(uname -s)" != "Linux" ]]; then
    warn "This tool needs Linux (ydotool/uinput). Detected: $(uname -s)."
    exit 1
fi

# ── distro detection ────────────────────────────────────────────────
ID=""; ID_LIKE=""
[[ -r /etc/os-release ]] && . /etc/os-release
FAMILY=" ${ID:-} ${ID_LIKE:-} "
say "Distro: ${PRETTY_NAME:-unknown} · session: ${XDG_SESSION_TYPE:-unknown}"

pkg_install() {
    case "$FAMILY" in
        *" debian "*|*" ubuntu "*) sudo apt update && sudo apt install -y "$@" ;;
        *" fedora "*|*" rhel "*|*" centos "*) sudo dnf install -y "$@" ;;
        *" arch "*|*" manjaro "*) sudo pacman -S --needed --noconfirm "$@" ;;
        *" opensuse "*|*" suse "*) sudo zypper install -y "$@" ;;
        *) warn "Unknown distro — please install manually: $*"; return 1 ;;
    esac
}

is_debian() { case "$FAMILY" in *" debian "*|*" ubuntu "*) return 0 ;; *) return 1 ;; esac; }

ydotool_modern() {
    command -v ydotool >/dev/null 2>&1 || return 1
    ydotool mousemove --help 2>&1 | grep -qi absolute
}

# ── 1. ydotool >= 1.0 (+ ydotoold) ─────────────────────────────────
if ydotool_modern && command -v ydotoold >/dev/null 2>&1; then
    ok "ydotool >= 1.0 with ydotoold already installed."
elif is_debian; then
    # Debian/Ubuntu apt ships only the incompatible 0.1.x — build v1.x.
    say "Building ydotool >= 1.0 from source (Debian/Ubuntu apt only has 0.1.x)…"
    pkg_install build-essential cmake scdoc git checkinstall
    src="$(mktemp -d)"
    # Full clone (not shallow) so `git describe` can resolve the latest tag.
    git clone https://github.com/ReimuNotMoe/ydotool "$src"
    cmake -B "$src/build" "$src"
    make -C "$src/build"
    # Package version from the actual checkout (latest tag + commits since), e.g.
    # "1.0.4+30+gab0561a"; sanitised for dpkg (no leading 'v', '-' → '+').
    pkgver="$(git -C "$src" describe --tags --always 2>/dev/null | sed 's/^v//; s/-/+/g')"
    [ -n "$pkgver" ] || pkgver="0+git$(date +%Y%m%d)"
    say "Building ydotool $pkgver"
    # Install via checkinstall so it's tracked by dpkg (clean uninstall with
    # `sudo dpkg -r ydotool`). --fstrans=no lets it follow the build's absolute
    # install paths; --default answers prompts non-interactively.
    if command -v checkinstall >/dev/null 2>&1; then
        ( cd "$src/build" && sudo checkinstall --fstrans=no --default \
            --pkgname=ydotool --pkgversion="$pkgver" --pkgsource="ydotool" \
            --backup=no --deldoc=yes make install )
    else
        warn "checkinstall unavailable — falling back to 'make install'."
        sudo make -C "$src/build" install
    fi
    sudo ldconfig || true
    rm -rf "$src"
    ok "ydotool installed to /usr/local/bin."
else
    say "Installing ydotool from your distro (v1.x)…"
    pkg_install ydotool || warn "Install ydotool manually, then re-run."
fi

# ── 2. screenshot backend (action verification) ─────────────────────
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" ]]; then
    if command -v grim >/dev/null 2>&1; then ok "grim present."; else
        say "Installing grim (Wayland screenshots)…"; pkg_install grim || warn "Install grim manually."
    fi
else
    say "X11 screenshots use the Python 'mss' package — install the extra with:"
    printf '      uv sync --extra input-control\n'
    command -v scrot >/dev/null 2>&1 || pkg_install scrot || true  # optional fallback
fi

# ── 3. /dev/uinput access ───────────────────────────────────────────
say "Granting /dev/uinput access (udev rule + 'input' group)…"
echo 'KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"' \
    | sudo tee /etc/udev/rules.d/99-uinput.rules >/dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger || true
echo uinput | sudo tee /etc/modules-load.d/uinput.conf >/dev/null
sudo modprobe uinput || true
if id -nG "$USER" | tr ' ' '\n' | grep -qx input; then
    ok "$USER is already in the 'input' group."
else
    sudo usermod -aG input "$USER"
    warn "Added $USER to 'input' — LOG OUT AND BACK IN for it to take effect."
fi

# ── 4. ydotoold user service (v1.x needs the daemon) ───────────────
if command -v ydotoold >/dev/null 2>&1; then
    say "Installing a ydotoold user service…"
    unitdir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    mkdir -p "$unitdir"
    ydotoold_bin="$(command -v ydotoold)"
    cat > "$unitdir/ydotoold.service" <<EOF
[Unit]
Description=ydotool daemon (leuk desktop control)

[Service]
ExecStart=$ydotoold_bin --socket-path=%t/.ydotool_socket --socket-perm=0600
Restart=always

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now ydotoold.service 2>/dev/null \
        && ok "ydotoold service enabled." \
        || warn "Could not start ydotoold yet (likely needs the 'input' group — re-login, then: systemctl --user restart ydotoold)."
    printf '      socket: %s/.ydotool_socket\n' "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
fi

echo
ok "Done. Next:"
printf '   1. Log out and back in (for the input-group membership).\n'
printf "   2. In leuk, enable the tool in /settings → General (saved to config.json).\n"
printf '   3. Check everything: leuk doctor\n'
