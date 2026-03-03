import os
from typing import List

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")


def _create_client() -> AzureOpenAI:
    """
    Create an Azure OpenAI client using API key authentication.
    """
    if not AZURE_OPENAI_ENDPOINT:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        raise ValueError("Missing AZURE_OPENAI_API_KEY")
    if not AZURE_OPENAI_API_VERSION:
        raise ValueError("Missing AZURE_OPENAI_API_VERSION")
    if not AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT:
        raise ValueError("Missing AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


_client = _create_client()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    Returns a list of vectors (one vector per input text).
    """
    resp = _client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
        input=texts,
    )
    return [item.embedding for item in resp.data]