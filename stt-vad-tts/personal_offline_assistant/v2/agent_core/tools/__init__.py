"""Tool layer. The ONLY code allowed to touch the network (weather, search, telegram).

Everything else in agent_core runs fully offline. The problem statement's rule:
reasoning stays on-device; online calls are lookups the user asked for.
"""
from agent_core.tools.router import ToolRouter

__all__ = ["ToolRouter"]
