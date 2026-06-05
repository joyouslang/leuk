[Home](../README.md) › [Tools](../tools.md) › Monitoring

# Monitoring tool

`src/leuk/tools/monitoring.py:MonitoringTool` gathers **read-only** data about the
host **without controlling it**. These observation-only capabilities used to live
inside the high-risk [input_control](input_control.md) tool — they're split out so
you can let the agent *look* at the machine without escalating to full
keyboard/mouse control.

Enable with the **Monitoring** toggle in `/settings → General` (or
`{"monitoring": {"enabled": true}}` / `LEUK_MONITORING_ENABLED=true`), and install
the screenshot deps:

```bash
uv sync --extra monitoring
```

## Actions

| action | result |
|--------|--------|
| `screenshot` | capture the **whole desktop/screen** as an image the model can see (HiDPI-downscaled like input_control — see [Multimodal](../multimodal.md)) |
| `geometry` | the screen resolution in pixels (the coordinate space a screenshot is captured in) |
| `system_info` | OS, hostname, CPU cores, load average, memory, disk, uptime |

Screenshots use `mss` on X11 and `grim` (or GNOME/KDE tools) on Wayland — see
[System Dependencies → Screenshots](../reference/system-dependencies.md#screenshots).
The capture + HiDPI-scaling helpers live in `src/leuk/host.py`, shared with
`input_control`.

## Safety

Everything here is read-only — it never injects input or changes the machine — so
it's far lower-risk than [input_control](input_control.md). A captured screenshot
can still contain sensitive on-screen content, so the tool is opt-in.

## See also

- [Tools](../tools.md) · [Input Control](input_control.md) · [Multimodal](../multimodal.md)
