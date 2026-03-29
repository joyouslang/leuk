"""ContainerRunner and ContainerSandbox: persistent Docker container isolation."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from typing import TYPE_CHECKING

from leuk.sandbox.mount_policy import validate_mounts

if TYPE_CHECKING:
    from leuk.config import SandboxConfig

logger = logging.getLogger(__name__)

_MAX_OUTPUT = 50_000


class ContainerRunner:
    """Low-level helper for managing Docker containers.

    Starts a long-lived container once per session and reuses it via
    ``docker exec`` for every subsequent command.  This preserves process
    state (env vars, working directory, installed packages) across calls
    and avoids the per-command container start-up overhead.
    """

    def __init__(self) -> None:
        self._last_used: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        session_id: str,
        mounts: list[tuple[str, str, bool]],
        image: str,
        resource_limits: dict[str, str],
    ) -> str:
        """Start a detached container and return its ID."""
        if not shutil.which("docker"):
            raise RuntimeError("Docker is not available")

        cmd = self._build_run_command(session_id, mounts, image, resource_limits)
        logger.debug("Starting container: %s", " ".join(cmd))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)

        if process.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"Failed to start container: {err}")

        container_id = stdout.decode().strip()
        self._last_used[container_id] = time.monotonic()
        logger.info("Started container %s for session %s", container_id[:12], session_id)
        return container_id

    async def exec(
        self,
        container_id: str,
        command: str,
        workdir: str | None,
        timeout: int,
    ) -> str:
        """Run a command inside an already-running container."""
        self._last_used[container_id] = time.monotonic()

        cmd = ["docker", "exec", "-i"]
        if workdir:
            cmd += ["-w", workdir]
        cmd += [container_id, "/bin/sh", "-c", command]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
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
            return f"[ERROR] Failed to exec in container: {exc}"

        return _format_output(stdout, stderr, process.returncode)

    async def stop(self, container_id: str) -> None:
        """Stop a running container."""
        self._last_used.pop(container_id, None)
        if not shutil.which("docker"):
            return
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "stop", "--time=5", container_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(process.communicate(), timeout=15)
            logger.info("Stopped container %s", container_id[:12])
        except Exception as exc:
            logger.warning("Failed to stop container %s: %s", container_id[:12], exc)

    async def stop_if_idle(self, container_id: str, idle_seconds: float) -> bool:
        """Stop the container if it has been idle for *idle_seconds*.

        Returns True if the container was stopped.
        """
        last = self._last_used.get(container_id)
        if last is None:
            return False
        if time.monotonic() - last >= idle_seconds:
            # Remove tracking entry first so callers don't see a stale entry
            # even if the concrete stop() implementation is replaced in tests.
            self._last_used.pop(container_id, None)
            await self.stop(container_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_run_command(
        self,
        session_id: str,
        mounts: list[tuple[str, str, bool]],
        image: str,
        resource_limits: dict[str, str],
    ) -> list[str]:
        uid = os.getuid()
        gid = os.getgid()

        cmd = [
            "docker", "run",
            "--rm",         # auto-remove on exit
            "-d",           # detached — keep running
            "-i",           # keep stdin open (required for docker exec)
            "--network=none",
            f"--user={uid}:{gid}",
            f"--name=leuk-{session_id[:12]}",
        ]

        if resource_limits.get("memory"):
            cmd += [f"--memory={resource_limits['memory']}"]
        if resource_limits.get("cpus"):
            cmd += [f"--cpus={resource_limits['cpus']}"]
        if resource_limits.get("pids"):
            cmd += [f"--pids-limit={resource_limits['pids']}"]

        for host_path, container_path, read_only in mounts:
            flag = (
                f"{host_path}:{container_path}:ro"
                if read_only
                else f"{host_path}:{container_path}"
            )
            cmd += ["-v", flag]

        cmd.append(image)
        # Keep the container alive with a blocking no-op
        cmd += ["tail", "-f", "/dev/null"]
        return cmd


class ContainerSandbox:
    """High-level sandbox that owns a single persistent container per session.

    The container is started lazily on the first ``execute()`` call and stopped
    explicitly by calling ``shutdown()``.

    Falls back gracefully when Docker is not available.
    """

    def __init__(self, config: SandboxConfig) -> None:
        self._config = config
        self._mounts = validate_mounts(config.allowed_mounts)
        self._runner = ContainerRunner()
        self._container_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ensure_started(self) -> str | None:
        """Lazily start the container on the first use.

        Returns the container ID, or None if Docker is unavailable.
        """
        if self._container_id is not None:
            return self._container_id
        if not shutil.which("docker"):
            return None
        try:
            # Use a stable session ID derived from object identity so the
            # container name is unique per ContainerSandbox instance.
            session_id = f"{id(self):016x}"
            self._container_id = await self._runner.start(
                session_id=session_id,
                mounts=self._mounts,
                image=self._config.image,
                resource_limits=self._config.resource_limits,
            )
        except RuntimeError as exc:
            logger.error("Could not start sandbox container: %s", exc)
            return None
        return self._container_id

    async def execute(
        self,
        command: str,
        workdir: str | None = None,
        timeout: int = 120,
    ) -> str:
        """Run *command* inside the container and return combined stdout/stderr."""
        container_id = await self.ensure_started()
        if container_id is None:
            return "[ERROR] Docker is not available — sandbox mode requires Docker"
        return await self._runner.exec(container_id, command, workdir, timeout)

    async def shutdown(self) -> None:
        """Stop the container, releasing all resources."""
        if self._container_id is not None:
            await self._runner.stop(self._container_id)
            self._container_id = None


def _format_output(stdout: bytes, stderr: bytes, returncode: int | None) -> str:
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
