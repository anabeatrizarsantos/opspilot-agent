from fastapi import FastAPI
from pydantic import BaseModel

from app.router import route
from app.llm_client import ask_llm
from app.main import create_ticket_from_message
from app.tools.ticketing import save_ticket
from app.rag import answer_with_rag

from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="OpsPilot API")

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Routes the request and executes the appropriate action.
    """

    decision = route(request.message)

    if decision.action == "answer":
        reply = answer_with_rag(request.message)
        return ChatResponse(reply=reply)

    elif decision.action == "create_ticket":
        ticket = create_ticket_from_message(request.message)
        path = save_ticket(ticket)
        return ChatResponse(reply=f"Ticket created at {path}")

    return ChatResponse(reply="Unable to process request.")