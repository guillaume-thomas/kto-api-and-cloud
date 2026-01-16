import os
import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from titanic.mcp_server.auth import token_manager

API_URL = os.getenv("TITANIC_API_URL", "http://titanic-api-service.{{ cookiecutter.developer_redhat_username }}-dev.svc.cluster.local:8080")


async def predict_survival(pclass: int, sex: str, sibsp: int, parch: int) -> str:
    """
    Prédit la survie d'un passager du Titanic.

    Args:
        pclass: Classe du billet (1, 2 ou 3)
        sex: Sexe ("male" ou "female")
        sibsp: Nombre de frères/sœurs/conjoints à bord
        parch: Nombre de parents/enfants à bord

    Returns:
        Prédiction de survie avec message et détails

    """

    return "Tool not implemented yet"


if __name__ == "__main__":
    print("toto")