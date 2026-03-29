"""Shell command execution tool."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from leuk.types import ToolSpec

if TYPE_CHECKING:
    from leuk.config import SandboxConfig

_MAX_OUTPUT = 50_000  # Truncate output beyond this many characters


class ShellTool:
    """Execute shell commands in a subprocess, optionally inside a Docker sandbox."""

    def __init__(self, sandbox: "SandboxConfig | None" = None) -> None:
        self._sandbox_config = sandbox
        # Lazily created — only instantiated when container mode is active.
        self._container_sandbox: Any = None

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

        if self._sandbox_config is not None and self._sandbox_config.mode == "container":
            return await self._execute_in_container(command, workdir, timeout)
        return await self._execute_local(command, workdir, timeout)

    async def shutdown_sandbox(self) -> None:
        """Stop the persistent container, if one was started."""
        if self._container_sandbox is not None:
            await self._container_sandbox.shutdown()
            self._container_sandbox = None

    # ------------------------------------------------------------------
    # Execution backends
    # ------------------------------------------------------------------

    async def _execute_local(
        self, command: str, workdir: str | None, timeout: int
    ) -> str:
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

        return _format_output(stdout, stderr, process.returncode, timeout)

    async def _execute_in_container(
        self, command: str, workdir: str | None, timeout: int
    ) -> str:
        from leuk.sandbox.container import ContainerSandbox

        if self._container_sandbox is None:
            self._container_sandbox = ContainerSandbox(self._sandbox_config)

        return await self._container_sandbox.execute(
            command, workdir=workdir, timeout=timeout
        )


def _format_output(
    stdout: bytes, stderr: bytes, returncode: int | None, timeout: int
) -> str:
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

    if returncode != 0:
        result = f"[exit code {returncode}]\n{result}"

    return result
