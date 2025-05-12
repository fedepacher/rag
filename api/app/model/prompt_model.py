from datetime import datetime
from typing import Optional


class Prompt:
    def __init__(self, input: str, date_in: Optional[datetime] = None, output: Optional[str] = None,
                 email: Optional[str] = None, status: Optional[str] = None, date_out: Optional[datetime] = None):
        self.input = input
        self.date_in = date_in or datetime.now()
        self.output = output
        self.email = email
        self.status = status
        self.date_out = date_out

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB insertion."""
        return {
            "input": self.input,
            "date_in": self.date_in,
            "output": self.output,
            "email": self.email,
            "status": self.status,
            "date_out": self.date_out,
        }
