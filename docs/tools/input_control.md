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

Coordinates are **full-resolution screen pixels** (origin top-left, `x` right,
`y` down). Call `geometry` for the exact width×height. Every pixel is addressable.

- **Exact pixel: `x`/`y`** — absolute full-resolution pixels, actuated 1:1.
- **Rough location: `xpct`/`ypct`, `0`–`100`** — percent of the screen, for a
  coarse target or a `zoom` centre.

`move` = absolute; `move_rel` = relative pixel deltas.

### Reaching an exact pixel: `zoom`

A vision model only ever sees a **downscaled** screenshot — we cap it, and the
model's own vision encoder shrinks it further — so it cannot resolve individual
pixels of, say, a 3840×2160 screen from the overview. The plain `screenshot` is
that downscaled overview, good only for *locating a region*. To click a precise
point:

1. `zoom` into the area — centre it with `xpct`/`ypct` (or `x`/`y`) plus a `zoom`
   factor (default 8). It returns a **magnified crop with a grid labelled in real
   screen coordinates** (requires Pillow, in the `input-control` extra).
2. Read the exact `x`,`y` off the grid.
3. `click` with those `x`,`y`.

Raise the `zoom` factor for finer reading. This is how the agent reaches any
single pixel regardless of how small the model's perception is.

## Actions

| action | args | effect |
|--------|------|--------|
| `geometry` | — | returns the full screen size `WxH px` |
| `screenshot` | — | downscaled overview PNG (locate regions; not pixel-exact) |
| `zoom` | `xpct`,`ypct` or `x`,`y` centre, `zoom` factor | magnified crop with a coordinate-labelled grid to read exact pixels |
| `move` | `xpct`,`ypct` or `x`,`y` | move pointer to absolute target |
| `move_rel` | `x`,`y` | move pointer by delta |
| `click`/`right_click`/`middle_click`/`double_click` | optional `xpct`,`ypct` or `x`,`y` | move (if given) then click |
| `mouse_down`/`mouse_up` | `button` | press/release a button |
| `scroll` | `direction` (up/down), `amount` | keyboard PageUp/PageDown scroll |
| `type` | `text`, optional `key_delay` (ms) | type a string |
| `key` | `key` (e.g. `ctrl+c`, `alt+Tab`, `super+l`) | press a combo |
| `key_down`/`key_up` | `key` | hold/release one key |

Pass `verify: true` on any action to attach **before *and* after** screenshots,
so the agent can compare the desktop state pre/post action and direct its next
move. (`geometry`/`screenshot` take no before-frame.)

## Verification

- After a **failure** or a **timeout**, a screenshot of the current desktop is
  attached automatically so the model can see the real state and recover (plus
  the before-frame when one was captured).
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
