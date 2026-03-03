import json

from app.llm_client import ask_llm
from app.schemas import RouteDecision


ROUTER_SYSTEM_PROMPT = """
You are a strict routing system for an operations support agent.

Your job is to decide what action should be taken.

Choose "create_ticket" ONLY when:
- The user explicitly asks to log, create, open, or submit a ticket
- The user clearly requests support or action

Choose "answer" when:
- The user is asking a question
- The user is making a general statement
- The user is describing an issue but not explicitly requesting support

Return ONLY valid JSON:
{"action": "answer"}
or
{"action": "create_ticket"}

No explanations.
No markdown.
No extra fields.
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