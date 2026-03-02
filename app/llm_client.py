import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file into the process
load_dotenv()

# Read Azure OpenAI configuration from environment variables
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


def _create_client() -> AzureOpenAI:
    """
    Create and return an authenticated Azure OpenAI client.

    The client is responsible for sending HTTP requests to the Azure OpenAI
    service using the provided credentials and endpoint.
    """
    if not API_KEY:
        raise ValueError("Missing AZURE_OPENAI_API_KEY in .env")

    if not ENDPOINT:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT in .env")

    if not DEPLOYMENT_NAME:
        raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT in .env")

    return AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
    )


# Initialize a single shared client (recommended pattern)
client = _create_client()


def ask_llm(message: str) -> str:
    """
    Send a user message to the LLM and return the assistant response.

    This function will become the central gateway for all agent reasoning.
    Every router/tool later will call this instead of calling the model directly.
    """

    response = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are OpsPilot, an operations support AI agent that helps classify requests and assist users.",
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        # temperature=0.2, this model doesn't support temperature
    )

    # Extract the assistant text from the API response
    return response.choices[0].message.content