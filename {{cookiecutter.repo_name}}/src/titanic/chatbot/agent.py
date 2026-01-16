import os
import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from fastmcp import Client



class ChatbotAgent:
    def __init__(self) -> None:
        mcp_server_host = os.getenv(
            "MCP_SERVER_HOST", "http://titanic-mcp-server.{{ cookiecutter.developer_redhat_username }}-dev.svc.cluster.local:8000"
        )

    async def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Appelle un tool MCP et extrait le résultat texte."""
        return ""

    async def chat_async(self, message: str) -> str:
        """Chat async qui charge les tools MCP et les utilise via le LLM."""
        return ""

    def chat(self, message: str) -> str:
        return asyncio.run(self.chat_async(message))
