import json

from app.llm_client import ask_llm
from app.schemas import Ticket
from app.tools.ticketing import save_ticket
from app.agent.orchestrator import run_agent


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
"""

def create_ticket_from_message(user_message: str) -> Ticket:
    """
    Ask the LLM to generate a structured ticket.
    Then validate it using the Ticket schema.
    """

    full_prompt = f"{TICKET_SYSTEM_PROMPT}\n\nUser request: {user_message}"

    raw_response = ask_llm(full_prompt)

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        raise ValueError("LLM did not return valid JSON for ticket generation.")
    
    try:
        ticket = Ticket.model_validate(data)
    except Exception as e:
        raise ValueError(f"Ticket validation failed: {e}")

    return ticket

def main():
    """
    OpsPilot v0.1 agent loop (routing + basic answering).

    Behavior:
    - If router decides "answer", we call the LLM normally and print the response.
    - If router decides "create_ticket", we only acknowledge for now (ticket creation comes next).
    """

    print("OpsPilot Agent started. Type 'exit' to stop.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            break

        result = run_agent(user_input)
        
        print(f"OpsPilot: {result.reply}")


if __name__ == "__main__":
    main()