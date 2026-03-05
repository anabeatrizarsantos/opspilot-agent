from app.rag import answer_with_rag

def search_kb(question: str) -> str:
    return answer_with_rag(question)