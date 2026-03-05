from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NextAction(str, Enum):
    ANSWER_WITH_KB = "answer_with_kb"
    CREATE_TICKET = "create_ticket"
    ASK_CLARIFYING = "ask_clarifying"
    ESCALATE_HUMAN = "escalate_human"


class AgentDecision(BaseModel):
    action: NextAction
    confidence: float = Field(ge=0.0, le=1.0)
    clarifying_question: Optional[str] = None


class AgentResponse(BaseModel):
    reply: str
    action: NextAction
    request_id: str
    tool_used: Optional[str] = None
    ticket_path: Optional[str] = None