"""Shell command execution tool."""

from __future__ import annotations

import asyncio
from typing import Any

from leuk.types import ToolSpec

_MAX_OUTPUT = 50_000  # Truncate output beyond this many characters


class ShellTool:
    """Execute shell commands in a subprocess."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell",
            description=(
                "Execute a shell command and return its stdout/stderr. "
                "Use this for running programs, installing packages, git operations, etc."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory (optional, defaults to cwd)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 120)",
                    },
                },
                "required": ["command"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        command = arguments["command"]
        workdir = arguments.get("workdir")
        timeout = arguments.get("timeout", 120)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            return f"[ERROR] Command timed out after {timeout}s"
        except OSError as exc:
            return f"[ERROR] Failed to execute command: {exc}"

        parts: list[str] = []
        if stdout:
            out = stdout.decode(errors="replace")
            if len(out) > _MAX_OUTPUT:
                out = out[:_MAX_OUTPUT] + f"\n... [truncated, {len(stdout)} bytes total]"
            parts.append(out)
        if stderr:
            err = stderr.decode(errors="replace")
            if err.strip():
                parts.append(f"[STDERR]\n{err}")

        result = "\n".join(parts) if parts else "(no output)"

        if process.returncode != 0:
            result = f"[exit code {process.returncode}]\n{result}"

        return result
