import json
from pathlib import Path
from datetime import datetime

from app.schemas import Ticket


# Directory where ticket files will be stored
TICKETS_DIR = Path("data/tickets")


def save_ticket(ticket: Ticket) -> Path:
    """
    Save a validated Ticket object as a JSON file on disk.
    Returns the file path.
    """

    # Create folder if it doesn't exist
    TICKETS_DIR.mkdir(parents=True, exist_ok=True)

    # Create timestamp-based filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"ticket_{timestamp}.json"
    file_path = TICKETS_DIR / filename

    # Convert Pydantic model to dict
    payload = ticket.model_dump()

    # Write JSON file
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return file_path