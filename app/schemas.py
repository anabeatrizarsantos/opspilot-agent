from typing import Literal
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """
    A small, strict schema for the router output.

    The router will read the user's message and choose ONE action.
    This schema forces the model output to be predictable and valid.
    """

    action: Literal["answer", "create_ticket"] = Field(
        ...,
        description="What the agent should do next for this user request.",
    )


class Ticket(BaseModel):
    """
    A structured support ticket.

    We will later ask the LLM to produce JSON that matches this schema,
    then validate it with Pydantic before saving it to disk.
    """

    title: str = Field(..., description="Short ticket title.")
    category: Literal["bug", "billing", "access", "feature_request", "other"] = Field(
        ...,
        description="Ticket category label.",
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        ...,
        description="Ticket priority level.",
    )
    summary: str = Field(..., description="One short paragraph describing the issue.")
    user_request: str = Field(..., description="The original user message.")