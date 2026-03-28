"""ContainerSandbox: runs shell commands inside a Docker container."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import TYPE_CHECKING

from leuk.sandbox.mount_policy import validate_mounts

if TYPE_CHECKING:
    from leuk.config import SandboxConfig

_MAX_OUTPUT = 50_000


class ContainerSandbox:
    """Execute commands inside a Docker container with bind mounts and resource limits.

    Falls back gracefully when Docker is not available by returning an error
    string rather than raising.
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._mounts = validate_mounts(config.allowed_mounts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 120,
    ) -> str:
        """Run *command* inside the container and return combined stdout/stderr."""
        if not shutil.which("docker"):
            return "[ERROR] Docker is not available — sandbox mode requires Docker"

        docker_cmd = self._build_docker_command(command, workdir)

        try:
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return f"[ERROR] Container command timed out after {timeout}s"
        except OSError as exc:
            return f"[ERROR] Failed to start container: {exc}"

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_docker_command(
        self, command: str, workdir: str | None
    ) -> list[str]:
        cfg = self._config
        uid = os.getuid()
        gid = os.getgid()

        cmd = [
            "docker", "run",
            "--rm",
            "--network=none",
            f"--user={uid}:{gid}",
        ]

        # Resource limits
        limits = cfg.resource_limits
        if limits.get("memory"):
            cmd += [f"--memory={limits['memory']}"]
        if limits.get("cpus"):
            cmd += [f"--cpus={limits['cpus']}"]
        if limits.get("pids"):
            cmd += [f"--pids-limit={limits['pids']}"]

        # Bind mounts from validated policy
        for host_path, container_path, read_only in self._mounts:
            flag = f"{host_path}:{container_path}:ro" if read_only else f"{host_path}:{container_path}"
            cmd += ["-v", flag]

        # Working directory inside container
        if workdir:
            cmd += ["-w", workdir]

        cmd.append(cfg.image)
        cmd += ["/bin/sh", "-c", command]

        return cmd
