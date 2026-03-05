import json
import uuid

from app.agent.prompts import DECISION_SYSTEM_PROMPT
from app.agent.schemas import AgentDecision, AgentResponse, NextAction
from app.agent.memory import get_history, append_message

from app.llm_client import ask_llm, ask_llm_messages
from app.tools.kb import search_kb
from app.tools.ticket_generator import create_ticket_from_message
from app.tools.ticketing import save_ticket


def _history_to_messages(history: list[tuple[str, str]]) -> list[dict[str, str]]:
    """
    Converts [("user","..."), ("assistant","...")] into:
    [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    """
    return [{"role": role, "content": content} for role, content in history]


def run_agent(message: str, session_id: str = "default") -> AgentResponse:
    request_id = str(uuid.uuid4())

    # 1) Get memory for this session
    history = get_history(session_id)

    # 2) Build messages = (history + current user message)
    messages = _history_to_messages(history)
    messages.append({"role": "user", "content": message})

    # 3) Ask LLM to decide next action (now with context)
    raw = ask_llm_messages(messages=messages, system_prompt=DECISION_SYSTEM_PROMPT)

    try:
        decision_dict = json.loads(raw)
        decision = AgentDecision.model_validate(decision_dict)
    except Exception:
        decision = AgentDecision(
            action=NextAction.ASK_CLARIFYING,
            confidence=0.2,
            clarifying_question="Can you clarify what you need help with?"
        )

    # 4) Execute action
    if decision.action == NextAction.ANSWER_WITH_KB:
        try:
            reply = search_kb(message)
            result = AgentResponse(
                reply=reply,
                action=decision.action,
                request_id=request_id,
                tool_used="kb.search",
            )
        except FileNotFoundError:
            result = AgentResponse(
                reply="My knowledge base is not indexed yet. Please run the KB indexer, or tell me if you want me to create a support ticket.",
                action=NextAction.ASK_CLARIFYING,
                request_id=request_id,
            )

    elif decision.action == NextAction.CREATE_TICKET:
        ticket = create_ticket_from_message(message)
        path = save_ticket(ticket)
        path_str = str(path)  # important: Path -> str
        result = AgentResponse(
            reply=f"Ticket created at {path_str}",
            action=decision.action,
            request_id=request_id,
            tool_used="ticket.create",
            ticket_path=path_str,
        )

    elif decision.action == NextAction.ESCALATE_HUMAN:
        result = AgentResponse(
            reply="This may require professional assistance. If you want, I can create a ticket with the details you provide.",
            action=decision.action,
            request_id=request_id,
        )

    else:
        # ASK_CLARIFYING
        question = decision.clarifying_question or "Can you share one more detail so I can help?"
        result = AgentResponse(
            reply=question,
            action=NextAction.ASK_CLARIFYING,
            request_id=request_id,
        )

    # 5) Save conversation to memory (user + assistant)
    append_message(session_id, "user", message)
    append_message(session_id, "assistant", result.reply)

    return result