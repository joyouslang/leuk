"""Tool system: capabilities the agent can use to interact with the environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from leuk.tools.base import Tool, ToolRegistry
from leuk.tools.browser import BrowserTool
from leuk.tools.file_edit import FileEditTool
from leuk.tools.file_read import FileReadTool
from leuk.tools.input_control import InputControlTool
from leuk.tools.local_llm import LocalLLMTool
from leuk.tools.memory_write import MemoryWriteTool
from leuk.tools.shell import ShellTool
from leuk.tools.sub_agent import SubAgentTool
from leuk.tools.web_fetch import WebFetchTool

if TYPE_CHECKING:
    from leuk.config import InputControlConfig, LocalLLMConfig, SandboxConfig
    from leuk.skills import SkillLoader


def create_default_registry(
    memory_dir: str = "~/.config/leuk/memory",
    memory_project_name: str = "",
    *,
    browser_enabled: bool = False,
    browser_headless: bool = False,
    sandbox: "SandboxConfig | None" = None,
    local_llm: "LocalLLMConfig | None" = None,
    input_control: "InputControlConfig | None" = None,
    skills_loader: "SkillLoader | None" = None,
) -> ToolRegistry:
    """Create a registry pre-loaded with all built-in tools."""
    registry = ToolRegistry()
    registry.register(ShellTool(sandbox=sandbox))
    registry.register(FileReadTool())
    registry.register(FileEditTool())
    registry.register(SubAgentTool())  # Manager injected later via set_manager()
    registry.register(WebFetchTool())
    registry.register(MemoryWriteTool(memory_dir=memory_dir, project_name=memory_project_name))
    if skills_loader is not None:
        from leuk.skills import SkillTool

        registry.register(SkillTool(skills_loader))
    if browser_enabled:
        registry.register(BrowserTool(headless=browser_headless))
    if local_llm is not None and local_llm.enabled:
        registry.register(
            LocalLLMTool(base_url=local_llm.base_url, default_model=local_llm.default_model)
        )
    if input_control is not None and input_control.enabled:
        registry.register(
            InputControlTool(verify=input_control.verify, ydotool_socket=input_control.ydotool_socket)
        )
    return registry


__all__ = [
    "Tool",
    "ToolRegistry",
    "create_default_registry",
    "BrowserTool",
    "InputControlTool",
    "LocalLLMTool",
    "MemoryWriteTool",
    "SubAgentTool",
    "WebFetchTool",
]
