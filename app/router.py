import json

from app.llm_client import ask_llm
from app.schemas import RouteDecision


ROUTER_SYSTEM_PROMPT = """
You are a strict routing system for an operations support agent.

Your job is to decide what action should be taken for each user request.

Valid actions:
- "answer" → when the user is asking a question or needs information.
- "create_ticket" → when the user is reporting a problem, requesting access, or asking to log something.

Return ONLY valid JSON in this exact format:
{"action": "answer"}
or
{"action": "create_ticket"}

Do not include explanations.
Do not include markdown.
Do not include extra fields.
"""


def route(user_message: str) -> RouteDecision:
    """
    Ask the LLM to choose the correct action for the given user message.
    Then validate the result using the RouteDecision schema.
    """

    # Combine instructions with the user input
    full_prompt = f"{ROUTER_SYSTEM_PROMPT}\n\nUser request: {user_message}"

    # Ask the LLM
    raw_response = ask_llm(full_prompt)

    # Convert string JSON to Python dict
    data = json.loads(raw_response)

    # Validate against schema (this will raise error if invalid)
    decision = RouteDecision.model_validate(data)

    return decision