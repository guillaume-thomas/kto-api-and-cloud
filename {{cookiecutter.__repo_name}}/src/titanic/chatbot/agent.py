import os
import asyncio
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
# TODO : Importer le client MCP depuis la librairie facilitant les échanges MCP

# TODO : Définir le système Prompt

class ChatbotAgent:
    def __init__(self) -> None:
        mcp_server_host = os.getenv(
            "MCP_SERVER_HOST", "http://titanic-mcp-server.{{ cookiecutter.developer_redhat_username }}-dev.svc.cluster.local:8000"
        )
        # TODO : Mettre en place dans un attribut de classe la configuration du client MCP en déclarant les servers mcp cibles
        # TODO : Mettre en place dans un attribut de classe l'abstraction du LLM de Langchain en tant que ChatOpenAI

    async def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Appelle un tool MCP et extrait le résultat texte."""
        # TODO : A implémenter : initialisattion asynchrone du client, appelle un tool spécifique
        return ""

    async def chat_async(self, message: str) -> str:
        """Chat async qui charge les tools MCP et les utilise via le LLM."""
        # TODO : A implémenter : initialisation asynchrone du client, récupération de la liste des tools
        # disponibles pour le client. Créer les abstractions des tools avec langchain et les binder avec le llm
        # TODO : Mettre en place la mécanique du chat avec les bons messages
        # TODO : Retourner le résultat du tool si c'est la réponse du llm, sinon, sa réponse générée.
        return ""

    def chat(self, message: str) -> str:
        return asyncio.run(self.chat_async(message))
