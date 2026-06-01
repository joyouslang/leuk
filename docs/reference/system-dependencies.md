[Home](../README.md) › [Configuration](../configuration.md) › System Dependencies

# System (OS / distribution) dependencies

Some optional features shell out to **system binaries** that are *not* installed
by `uv`/pip. If a feature can't find its dependency it now reports a precise
reason; this page lists what to install.

> **Just run `leuk doctor`.** It checks every optional feature's requirements for
> your distro and display server, prints exact copy-pasteable fix commands, and
> tells you how to enable each one. The same report is available as `/doctor` in
> the REPL. For desktop control specifically, `bash scripts/setup-input-control.sh`
> automates the whole setup.

| Feature | System deps | Python extra |
|---------|-------------|--------------|
| [Voice](#voice) | *none* (half-duplex; no audio-server setup) | `voice` |
| [Desktop control](#desktop-control) | **ydotool ≥ 1.0** + `ydotoold` + `/dev/uinput` access | `input-control` |
| [Screenshots](#screenshots) | `grim` (Wayland) / `scrot`·`maim` (X11) | `input-control` (`mss`) |
| [Browser](#browser) | Chromium (via `playwright install`) | `browser` |

---

## Voice

[Voice](../voice.md) needs **no system dependencies** beyond the `voice` extra.
Input is **half-duplex**: the mic is paused while the agent speaks (TTS) and
resumed afterwards, so there's no speaker→mic feedback loop and no audio-server
(PipeWire/PulseAudio echo-cancel) configuration to set up. Use **headphones** if
you want to talk while the agent is speaking.

---

## Desktop control

The [`input_control`](../tools/input_control.md) tool injects keyboard/mouse via
**ydotool** (kernel `uinput`), which works under **X11 and Wayland**.

**Fastest path — run the setup script** (idempotent; installs ydotool ≥ 1.0,
grants `/dev/uinput` access, and starts `ydotoold` as a user service):

```bash
bash scripts/setup-input-control.sh
leuk doctor          # verify
```

### Manual setup

leuk requires **ydotool ≥ 1.0**. The 1.x CLI is what the tool drives (absolute
mouse positioning, hex click codes, keycode syntax); the legacy **0.1.x** shipped
by Debian/Ubuntu apt has an incompatible CLI — it rejects `mousemove --absolute`
*while exiting 0*, so actions silently do nothing. leuk detects 0.1.x and returns
a clear `[ERROR]` instead of pretending to work.

```bash
# Fedora / Arch / openSUSE ship ydotool 1.x:
sudo dnf install ydotool          # Fedora
sudo pacman -S ydotool            # Arch
sudo zypper install ydotool       # openSUSE

# Debian/Ubuntu apt has only 0.1.x — build 1.x from source:
sudo apt install -y build-essential cmake scdoc git
git clone https://github.com/ReimuNotMoe/ydotool /tmp/ydotool
cmake -B /tmp/ydotool/build /tmp/ydotool && make -C /tmp/ydotool/build
sudo make -C /tmp/ydotool/build install

# Allow your user to use /dev/uinput (one-time, then re-login):
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | \
    sudo tee /etc/udev/rules.d/99-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo modprobe uinput
sudo usermod -aG input "$USER"    # log out and back in

# Run the daemon (v1.x needs it) — a user service is best:
ydotoold &                        # quick test
```

**The `ydotoold` daemon must be running** — `ydotool` 1.x injects through it, and
if it's down the client often prints a connection warning yet still exits 0, so
the action would *appear* to succeed while doing nothing. leuk guards against this:
it refuses 0.1.x, requires `ydotoold` to be installed, and checks the daemon
socket up front (treating a "failed to connect" warning as an error) — returning a
clear `[ERROR]` rather than silently no-op'ing. The socket is looked for at
`$XDG_RUNTIME_DIR/.ydotool_socket`, `/run/user/<uid>/.ydotool_socket`, then
`/tmp/.ydotool_socket`; if `ydotoold` runs elsewhere, set
`LEUK_INPUT_CONTROL_YDOTOOL_SOCKET` (exported as `YDOTOOL_SOCKET`).

Then enable the tool in leuk: **`/settings → General`** (saved to `config.json`).

---

## Screenshots

Screenshots are used by `input_control` for action verification, by the browser
tool, and in channel approval previews. leuk tries several backends and reports
which ones failed:

### X11

- `mss` (pip, in `leuk[input-control]`) — preferred.
- Fallbacks: `scrot`, `maim`, or ImageMagick `import`.

```bash
uv sync --extra input-control     # installs mss
sudo apt install scrot            # optional fallback
```

### Wayland

- `grim` — works on **wlroots** compositors (Sway, Hyprland, river).
- `gnome-screenshot` — GNOME.
- `spectacle` — KDE.

```bash
sudo apt install grim             # wlroots
sudo apt install gnome-screenshot # GNOME
sudo apt install kde-spectacle    # KDE (package name varies)
```

> On GNOME/KDE Wayland, `grim` won't work (no wlroots screencopy); install the
> matching desktop tool instead. leuk's error message lists exactly which
> backends it tried and why each failed.

---

## Browser

The [browser tool](../tools/browser.md) needs Chromium installed by Playwright:

```bash
uv sync --extra browser
uv run playwright install chromium
# (Playwright may prompt for system libs: `playwright install-deps`)
```

## See also

- [Getting Started](../getting-started.md) · [Voice](../voice.md) · [Input Control](../tools/input_control.md) · [Environment Variables](environment.md)
