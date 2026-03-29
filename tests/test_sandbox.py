"""Tests for container sandbox: ContainerRunner, ContainerSandbox, mount policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from leuk.sandbox.mount_policy import validate_mounts, _is_blocked
from leuk.sandbox.container import ContainerSandbox, ContainerRunner, _format_output
from leuk.config import SandboxConfig


# ── mount_policy ────────────────────────────────────────────────────


class TestIsBlocked:
    def test_blocks_ssh_dir(self):
        assert _is_blocked(Path.home() / ".ssh")

    def test_blocks_ssh_subdir(self):
        assert _is_blocked(Path.home() / ".ssh" / "id_rsa")

    def test_blocks_gnupg(self):
        assert _is_blocked(Path.home() / ".gnupg")

    def test_blocks_aws(self):
        assert _is_blocked(Path.home() / ".aws" / "credentials")

    def test_blocks_kube(self):
        assert _is_blocked(Path.home() / ".kube" / "config")

    def test_blocks_docker_config(self):
        assert _is_blocked(Path.home() / ".docker")

    def test_blocks_leuk_config(self):
        assert _is_blocked(Path.home() / ".config" / "leuk")

    def test_blocks_env_file_by_name(self):
        assert _is_blocked(Path("/some/project/.env"))

    def test_blocks_pem_by_suffix(self):
        assert _is_blocked(Path("/certs/server.pem"))

    def test_blocks_key_by_suffix(self):
        assert _is_blocked(Path("/certs/server.key"))

    def test_blocks_p12_by_suffix(self):
        assert _is_blocked(Path("/certs/bundle.p12"))

    def test_allows_normal_project_dir(self, tmp_path: Path):
        assert not _is_blocked(tmp_path / "src")

    def test_allows_home_documents(self):
        assert not _is_blocked(Path.home() / "Documents")


class TestValidateMounts:
    def test_simple_path(self, tmp_path: Path):
        result = validate_mounts([str(tmp_path)])
        assert len(result) == 1
        host, container, ro = result[0]
        assert host == str(tmp_path)
        assert ro is True  # default is read-only

    def test_rw_flag(self, tmp_path: Path):
        result = validate_mounts([f"{tmp_path}:rw"])
        _, _, ro = result[0]
        assert ro is False

    def test_explicit_ro_flag(self, tmp_path: Path):
        result = validate_mounts([f"{tmp_path}:ro"])
        _, _, ro = result[0]
        assert ro is True

    def test_custom_container_path(self, tmp_path: Path):
        result = validate_mounts([f"{tmp_path}:/workspace"])
        _, container, _ = result[0]
        assert container == "/workspace"

    def test_custom_container_path_rw(self, tmp_path: Path):
        result = validate_mounts([f"{tmp_path}:/workspace:rw"])
        _, container, ro = result[0]
        assert container == "/workspace"
        assert ro is False

    def test_blocks_ssh_raises(self):
        ssh_dir = str(Path.home() / ".ssh")
        with pytest.raises(ValueError, match="sensitive credentials"):
            validate_mounts([ssh_dir])

    def test_blocks_env_file_raises(self, tmp_path: Path):
        env_file = tmp_path / ".env"
        env_file.write_text("SECRET=x")
        with pytest.raises(ValueError, match="sensitive credentials"):
            validate_mounts([str(env_file)])

    def test_empty_list(self):
        assert validate_mounts([]) == []


# ── _format_output ──────────────────────────────────────────────────


class TestFormatOutput:
    def test_stdout_only(self):
        result = _format_output(b"hello\n", b"", 0)
        assert "hello" in result
        assert "[exit code" not in result

    def test_stderr_shown(self):
        result = _format_output(b"", b"oops\n", 1)
        assert "[STDERR]" in result
        assert "oops" in result
        assert "[exit code 1]" in result

    def test_nonzero_exit(self):
        result = _format_output(b"out", b"", 2)
        assert "[exit code 2]" in result

    def test_empty_output(self):
        result = _format_output(b"", b"", 0)
        assert result == "(no output)"

    def test_truncation(self):
        big = b"x" * 60_000
        result = _format_output(big, b"", 0)
        assert "truncated" in result

    def test_empty_stderr_not_shown(self):
        result = _format_output(b"hi", b"   \n  ", 0)
        assert "[STDERR]" not in result


# ── ContainerRunner ─────────────────────────────────────────────────


class TestContainerRunner:
    def test_build_run_command_includes_network_none(self):
        runner = ContainerRunner()
        cmd = runner._build_run_command(
            session_id="abc123",
            mounts=[],
            image="leuk-sandbox:latest",
            resource_limits={"memory": "512m", "cpus": "1.0", "pids": "256"},
        )
        assert "--network=none" in cmd
        assert "-d" in cmd
        assert "-i" in cmd
        assert "--rm" in cmd

    def test_build_run_command_applies_resource_limits(self):
        runner = ContainerRunner()
        cmd = runner._build_run_command(
            session_id="abc123",
            mounts=[],
            image="leuk-sandbox:latest",
            resource_limits={"memory": "256m", "cpus": "0.5", "pids": "128"},
        )
        assert "--memory=256m" in cmd
        assert "--cpus=0.5" in cmd
        assert "--pids-limit=128" in cmd

    def test_build_run_command_includes_mounts(self, tmp_path: Path):
        runner = ContainerRunner()
        cmd = runner._build_run_command(
            session_id="abc123",
            mounts=[(str(tmp_path), "/workspace", True)],
            image="leuk-sandbox:latest",
            resource_limits={},
        )
        assert "-v" in cmd
        assert f"{tmp_path}:/workspace:ro" in cmd

    def test_build_run_command_rw_mount(self, tmp_path: Path):
        runner = ContainerRunner()
        cmd = runner._build_run_command(
            session_id="abc123",
            mounts=[(str(tmp_path), "/workspace", False)],
            image="leuk-sandbox:latest",
            resource_limits={},
        )
        assert f"{tmp_path}:/workspace" in cmd
        assert f"{tmp_path}:/workspace:ro" not in cmd

    def test_idle_tracking(self):
        import time
        runner = ContainerRunner()
        runner._last_used["cid"] = time.monotonic() - 10
        # Not stopped yet (no docker), but the logic is testable via the dict
        assert "cid" in runner._last_used

    @pytest.mark.asyncio
    async def test_stop_if_idle_removes_entry(self):
        import time
        runner = ContainerRunner()
        runner._last_used["cid"] = time.monotonic() - 9999

        # Patch docker stop to succeed immediately
        async def _fake_stop(cid):
            pass

        runner.stop = _fake_stop
        stopped = await runner.stop_if_idle("cid", idle_seconds=1.0)
        assert stopped is True
        assert "cid" not in runner._last_used

    @pytest.mark.asyncio
    async def test_stop_if_idle_not_yet(self):
        import time
        runner = ContainerRunner()
        runner._last_used["cid"] = time.monotonic()
        stopped = await runner.stop_if_idle("cid", idle_seconds=9999.0)
        assert stopped is False


# ── ContainerSandbox ────────────────────────────────────────────────


class TestContainerSandbox:
    def _make_config(self, **kwargs) -> SandboxConfig:
        defaults = {
            "mode": "container",
            "image": "leuk-sandbox:latest",
            "allowed_mounts": [],
            "resource_limits": {"memory": "512m", "cpus": "1.0", "pids": "256"},
        }
        defaults.update(kwargs)
        return SandboxConfig(**defaults)

    @pytest.mark.asyncio
    async def test_returns_error_when_no_docker(self, tmp_path: Path):
        cfg = self._make_config()
        sandbox = ContainerSandbox(cfg)
        with patch("shutil.which", return_value=None):
            result = await sandbox.execute("echo hi")
        assert "[ERROR]" in result
        assert "Docker" in result

    @pytest.mark.asyncio
    async def test_ensure_started_returns_none_without_docker(self):
        cfg = self._make_config()
        sandbox = ContainerSandbox(cfg)
        with patch("shutil.which", return_value=None):
            cid = await sandbox.ensure_started()
        assert cid is None

    @pytest.mark.asyncio
    async def test_container_started_lazily(self):
        """ContainerSandbox starts the container only on first execute()."""
        cfg = self._make_config()
        sandbox = ContainerSandbox(cfg)

        started_calls = []

        async def _fake_start(session_id, mounts, image, resource_limits):
            started_calls.append(session_id)
            return "fake-container-id"

        async def _fake_exec(container_id, command, workdir, timeout):
            return f"ran: {command}"

        sandbox._runner.start = _fake_start
        sandbox._runner.exec = _fake_exec

        with patch("shutil.which", return_value="/usr/bin/docker"):
            result1 = await sandbox.execute("echo a")
            result2 = await sandbox.execute("echo b")

        # Container started exactly once
        assert len(started_calls) == 1
        assert result1 == "ran: echo a"
        assert result2 == "ran: echo b"
        assert sandbox._container_id == "fake-container-id"

    @pytest.mark.asyncio
    async def test_shutdown_stops_container(self):
        cfg = self._make_config()
        sandbox = ContainerSandbox(cfg)
        sandbox._container_id = "existing-cid"

        stopped = []

        async def _fake_stop(cid):
            stopped.append(cid)

        sandbox._runner.stop = _fake_stop
        await sandbox.shutdown()
        assert stopped == ["existing-cid"]
        assert sandbox._container_id is None

    @pytest.mark.asyncio
    async def test_shutdown_noop_when_not_started(self):
        cfg = self._make_config()
        sandbox = ContainerSandbox(cfg)
        # Should not raise
        await sandbox.shutdown()

    @pytest.mark.asyncio
    async def test_start_failure_returns_error(self):
        cfg = self._make_config()
        sandbox = ContainerSandbox(cfg)

        async def _fail_start(*args, **kwargs):
            raise RuntimeError("image not found")

        sandbox._runner.start = _fail_start

        with patch("shutil.which", return_value="/usr/bin/docker"):
            result = await sandbox.execute("echo hi")
        assert "[ERROR]" in result


# ── ShellTool sandbox integration ───────────────────────────────────


class TestShellToolSandboxIntegration:
    @pytest.mark.asyncio
    async def test_sandbox_mode_none_runs_locally(self):
        """With sandbox mode 'none', shell runs on the host."""
        from leuk.tools.shell import ShellTool

        tool = ShellTool(sandbox=None)
        result = await tool.execute({"command": "echo local"})
        assert "local" in result

    @pytest.mark.asyncio
    async def test_sandbox_mode_container_uses_sandbox(self):
        """With sandbox mode 'container', shell delegates to ContainerSandbox."""
        from leuk.tools.shell import ShellTool

        cfg = SandboxConfig(
            mode="container",
            image="leuk-sandbox:latest",
            allowed_mounts=[],
            resource_limits={},
        )
        tool = ShellTool(sandbox=cfg)

        call_log = []

        # Patching at class level means the function receives `self` as first arg.
        async def _fake_execute(self_arg, command, workdir=None, timeout=120):
            call_log.append(command)
            return f"container: {command}"

        with patch("leuk.sandbox.container.ContainerSandbox.execute", new=_fake_execute):
            result = await tool.execute({"command": "ls /workspace"})

        assert call_log == ["ls /workspace"]
        assert "container:" in result

    @pytest.mark.asyncio
    async def test_same_sandbox_instance_reused(self):
        """ShellTool reuses the same ContainerSandbox across multiple calls."""
        from leuk.tools.shell import ShellTool

        cfg = SandboxConfig(
            mode="container",
            image="leuk-sandbox:latest",
            allowed_mounts=[],
            resource_limits={},
        )
        tool = ShellTool(sandbox=cfg)

        async def _fake_execute(self_arg, command, workdir=None, timeout=120):
            return f"ok: {command}"

        with patch("leuk.sandbox.container.ContainerSandbox.execute", new=_fake_execute):
            await tool.execute({"command": "echo 1"})
            first_sandbox = tool._container_sandbox
            await tool.execute({"command": "echo 2"})
            second_sandbox = tool._container_sandbox

        assert first_sandbox is second_sandbox

    @pytest.mark.asyncio
    async def test_shutdown_sandbox_clears_instance(self):
        """shutdown_sandbox() stops the container and clears the reference."""
        from leuk.tools.shell import ShellTool

        cfg = SandboxConfig(
            mode="container",
            image="leuk-sandbox:latest",
            allowed_mounts=[],
            resource_limits={},
        )
        tool = ShellTool(sandbox=cfg)

        shutdown_called = []

        async def _fake_execute(self_arg, command, workdir=None, timeout=120):
            return "ok"

        async def _fake_shutdown():
            shutdown_called.append(True)

        with patch("leuk.sandbox.container.ContainerSandbox.execute", new=_fake_execute):
            await tool.execute({"command": "echo hi"})

        tool._container_sandbox.shutdown = _fake_shutdown
        await tool.shutdown_sandbox()
        assert shutdown_called == [True]
        assert tool._container_sandbox is None
