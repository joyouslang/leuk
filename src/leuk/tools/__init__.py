"""Tool system: capabilities the agent can use to interact with the environment."""

from __future__ import annotations

from leuk.tools.base import Tool, ToolRegistry
from leuk.tools.file_edit import FileEditTool
from leuk.tools.file_read import FileReadTool
from leuk.tools.local_llm import LocalLLMTool
from leuk.tools.memory_write import MemoryWriteTool
from leuk.tools.shell import ShellTool
from leuk.tools.sub_agent import SubAgentTool
from leuk.tools.web_fetch import WebFetchTool


def create_default_registry(
    memory_dir: str = "~/.config/leuk/memory",
    memory_project_name: str = "",
) -> ToolRegistry:
    """Create a registry pre-loaded with all built-in tools."""
    registry = ToolRegistry()
    registry.register(ShellTool())
    registry.register(FileReadTool())
    registry.register(FileEditTool())
    registry.register(SubAgentTool())  # Manager injected later via set_manager()
    registry.register(WebFetchTool())
    registry.register(MemoryWriteTool(memory_dir=memory_dir, project_name=memory_project_name))
    return registry


__all__ = [
    "Tool",
    "ToolRegistry",
    "create_default_registry",
    "LocalLLMTool",
    "MemoryWriteTool",
    "SubAgentTool",
    "WebFetchTool",
]
