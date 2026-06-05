"""Read-only host monitoring: screenshots, screen geometry, and system info.

These observation-only capabilities used to live inside the high-risk
``input_control`` tool. They gather data from the host **without** writing to or
controlling it, so they are exposed here as a separate low-risk tool — you can
enable monitoring without escalating to full desktop control.
"""

from __future__ import annotations

from typing import Any

from leuk import host
from leuk.types import ToolSpec


class MonitoringTool:
    """Observe the host: capture the screen, read its geometry, and report system info."""

    name = "monitoring"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=(
                "Read-only host monitoring — gather data about the machine without "
                "controlling it.\n"
                "ACTIONS:\n"
                "- 'screenshot': capture the user's desktop/screen as an image you can "
                "see (the whole screen, not just a browser page).\n"
                "- 'geometry': the screen resolution in pixels (the coordinate space a "
                "screenshot is captured in).\n"
                "- 'system_info': OS, CPU, memory, disk, load average and uptime."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["screenshot", "geometry", "system_info"],
                    }
                },
                "required": ["action"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        action = arguments.get("action")
        if action == "screenshot":
            size, _reason = host.screen_size()
            scale = (
                host.compute_scale(size[0], size[1])
                if size and host.pil_available()
                else 1.0
            )
            tag, reason = host.screenshot_tag(scale)
            if tag is None:
                return f"[ERROR] screenshot unavailable: {reason}"
            return tag
        if action == "geometry":
            size, reason = host.screen_size()
            if size is None:
                return f"[ERROR] could not determine screen geometry: {reason}"
            scale = host.compute_scale(size[0], size[1]) if host.pil_available() else 1.0
            lw, lh = max(1, round(size[0] * scale)), max(1, round(size[1] * scale))
            return f"screen: {lw}x{lh} px"
        if action == "system_info":
            return host.system_info()
        return f"[ERROR] unknown action {action!r} (use screenshot/geometry/system_info)"
