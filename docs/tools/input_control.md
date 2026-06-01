[Home](../README.md) › [Tools](../tools.md) › Input Control

# `input_control` tool — desktop keyboard/mouse control

Controls the real keyboard and mouse on Linux via **ydotool** (kernel
`uinput`), so it works identically under **X11 and Wayland**. Screenshots
(X11 via `mss`, Wayland via `grim`) are used to verify the effect of actions.

This tool is **high risk** — it drives the user's actual desktop. It is
disabled by default (`input_control.enabled`), always requires approval unless
the dedicated desktop auto-approve switch is on, and on non-local channels the
approval request carries before/after screenshots.

## System setup

Easiest: run the setup script, then verify with `leuk doctor`:

```bash
bash scripts/setup-input-control.sh   # installs ydotool ≥1.0, uinput access, daemon
leuk doctor                           # confirm it's all green
```

Requires **ydotool ≥ 1.0** — the legacy 0.1.x on Debian/Ubuntu apt has an
incompatible CLI and silently no-ops (leuk detects it and errors clearly). Manual
steps and the from-source build for Debian/Ubuntu are in
[System Dependencies → Desktop control](../reference/system-dependencies.md#desktop-control).

`mss` (X11 screenshots) and `Pillow` (HiDPI downscaling, below) come with
`leuk[input-control]`; Wayland uses `grim`. If `ydotoold` runs on a non-default
socket, set `LEUK_INPUT_CONTROL_YDOTOOL_SOCKET` (exported as `YDOTOOL_SOCKET`).

## Coordinate system

- Absolute **pixels**, origin `(0,0)` at the **top-left** of the screen;
  `x` grows right, `y` grows down.
- Always call `geometry` first, and derive click targets from a `screenshot`
  taken at that same resolution. The coordinate you read off the screenshot is
  the one to pass back — the tool maps it to real hardware pixels.
- `move` = absolute; `move_rel` = relative deltas.

### HiDPI / 4K scaling

Vision APIs downscale large screenshots before the model sees them (Anthropic
caps the long edge at ~1568px). On a 4K display that means the model reasons over
a shrunken image, so raw-pixel coordinates land at a fraction of the intended
spot and *nothing happens*. To avoid this the tool presents a **consistent
downscaled space** (long edge ≈ 1366px, WXGA): `geometry` reports that scaled
size, `screenshot` is captured at it, and `move`/`click` coordinates are scaled
back up to physical pixels automatically. This needs **Pillow** (in the
`input-control` extra); without it scaling is disabled and screenshots/coords
stay at native resolution.

## Actions

| action | args | effect |
|--------|------|--------|
| `geometry` | — | returns `screen: WxH px` |
| `screenshot` | — | returns a `[screenshot:…]` PNG |
| `move` | `x`,`y` | move pointer to absolute pixel |
| `move_rel` | `x`,`y` | move pointer by delta |
| `click`/`right_click`/`middle_click`/`double_click` | optional `x`,`y` | move (if given) then click |
| `mouse_down`/`mouse_up` | `button` | press/release a button |
| `scroll` | `direction` (up/down), `amount` | keyboard PageUp/PageDown scroll |
| `type` | `text`, optional `key_delay` (ms) | type a string |
| `key` | `key` (e.g. `ctrl+c`, `alt+Tab`, `super+l`) | press a combo |
| `key_down`/`key_up` | `key` | hold/release one key |

Pass `verify: true` on any action to attach a screenshot of the result.

## Verification

- After a **failure** or a **timeout**, a screenshot of the current desktop is
  attached automatically so the model can see the real state and recover.
- `input_control.verify` baseline: `on_failure` (default), `each_action`, or
  `never`.
- In **auto-approve** mode the agent should verify each step and **escalate
  risky/irreversible actions** (closing/deleting, submitting forms, sending,
  payments, system settings) for explicit user approval; routine reversible
  actions self-verify and continue.

## ydotool command mapping (maintainer reference)

| tool action | ydotool |
|-------------|---------|
| `move` | `ydotool mousemove --absolute -x X -y Y` |
| `move_rel` | `ydotool mousemove -x DX -y DY` |
| `click` left/right/middle | `ydotool click 0xC0` / `0xC1` / `0xC2` |
| `mouse_down`/`mouse_up` | `ydotool click 0x40`/`0x80` (left; +1/+2 for right/middle) |
| `type` | `ydotool type [--key-delay MS] "text"` |
| `key ctrl+c` | `ydotool key 29:1 46:1 46:0 29:0` (codes pressed then released in reverse) |

Key names map to evdev keycodes (see `KEYCODES` in
`src/leuk/tools/input_control.py`): letters `a-z`, digits `0-9`, modifiers
`ctrl/shift/alt/super`, `enter esc tab space backspace delete insert`,
`home end pageup pagedown`, `up/down/left/right`, `f1-f12`, and common
punctuation. Keep this table and the tool's `KEYCODES`/`spec.description` in
sync.

## See also

- [Tools](../tools.md) · [Safety & Approvals](../safety.md) · [Multimodal](../multimodal.md) · [Channels](../channels.md)
