import json

from app.llm_client import ask_llm
from app.schemas import Ticket

TICKET_SYSTEM_PROMPT = """
You are a strict support ticket generator.

You must return ONLY valid JSON in this exact format:

{
  "title": string,
  "category": "bug" | "billing" | "access" | "feature_request" | "other",
  "priority": "low" | "medium" | "high" | "urgent",
  "summary": string,
  "user_request": string
}

Do not include explanations.
Do not include markdown.
Do not include extra fields.
""".strip()


def create_ticket_from_message(user_message: str) -> Ticket:
    raw_response = ask_llm(user_message, system_prompt=TICKET_SYSTEM_PROMPT)

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON for ticket generation.")

    return Ticket.model_validate(data)