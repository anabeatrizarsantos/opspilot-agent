from collections import deque
from typing import Deque, Dict, List, Tuple

# session_id -> deque of (role, content)
_MEMORY: Dict[str, Deque[Tuple[str, str]]] = {}

MAX_TURNS = 10  # 10 mensagens no total (user+assistant)

def get_history(session_id: str) -> List[Tuple[str, str]]:
    dq = _MEMORY.get(session_id)
    if not dq:
        return []
    return list(dq)

def append_message(session_id: str, role: str, content: str) -> None:
    if session_id not in _MEMORY:
        _MEMORY[session_id] = deque(maxlen=MAX_TURNS)
    _MEMORY[session_id].append((role, content))