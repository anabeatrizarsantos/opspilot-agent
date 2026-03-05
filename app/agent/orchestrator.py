import json
import uuid

from app.llm_client import ask_llm
from app.agent.prompts import DECISION_SYSTEM_PROMPT
from app.agent.schemas import AgentDecision, AgentResponse, NextAction
from app.tools.kb import search_kb
from app.main import create_ticket_from_message
from app.tools.ticketing import save_ticket


def run_agent(message: str) -> AgentResponse:
    request_id = str(uuid.uuid4())

    raw = ask_llm(message, system_prompt=DECISION_SYSTEM_PROMPT)

    try:
        decision_dict = json.loads(raw)
        decision = AgentDecision.model_validate(decision_dict)
    except Exception:
        # fallback seguro se o modelo não devolver JSON perfeito
        decision = AgentDecision(action=NextAction.ASK_CLARIFYING, confidence=0.2,
                                clarifying_question="Can you clarify what you need help with?")

    if decision.action == NextAction.ANSWER_WITH_KB:
        try:
            reply = search_kb(message)
            return AgentResponse(
                reply=reply,
                action=decision.action,
                request_id=request_id,
                tool_used="kb.search",
            )
        except FileNotFoundError:
            return AgentResponse(
                reply="My knowledge base is not indexed yet. Please run the KB indexer, or tell me if you want me to create a support ticket.",
                action=NextAction.ASK_CLARIFYING,
                request_id=request_id,
            )

    if decision.action == NextAction.CREATE_TICKET:
        ticket = create_ticket_from_message(message)
        path = save_ticket(ticket)
        path_str = str(path)
        return AgentResponse(
            reply=f"Ticket created at {path_str}",
            action=decision.action,
            request_id=request_id,
            tool_used="ticket.create",
            ticket_path=path_str,
        )

    if decision.action == NextAction.ESCALATE_HUMAN:
        return AgentResponse(
            reply="This may require professional assistance. If you want, I can create a ticket with the details you provide (what happened, when, where, urgency).",
            action=decision.action,
            request_id=request_id,
        )

    # ASK_CLARIFYING (default)
    question = decision.clarifying_question or "Can you share one more detail so I can help?"
    return AgentResponse(
        reply=question,
        action=NextAction.ASK_CLARIFYING,
        request_id=request_id,
    )