#desde api/app/ archivo de igual nombre (comento las lineas que cambio)
from datetime import datetime
from typing import Optional


class Prompt:
    def __init__(self, input: str, date: Optional[datetime] = None, output: Optional[str] = None,
                 context: Optional[str] = None, email: Optional[str] = None, status: Optional[str] = None):
        self.input = input
        self.date = date or datetime.now()
        self.output = output
        self.context = context
        self.email = email
        self.status = status

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB insertion."""
        return {
            "input": self.input,
            "date": self.date,
            "output": self.output,
            "context": self.context,
            "email": self.email,
            "status": self.status
        }
