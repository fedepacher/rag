from datetime import datetime
from typing import Optional


class Prompt:
    def __init__(self, input: str, date: Optional[datetime] = None, output: Optional[str] = None,
                 context: Optional[str] = None, email: Optional[str] = None):
        self.input = input
        self.date = date or datetime.now()
        self.output = output
        self.context = context
        self.email = email

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB insertion."""
        return {
            "input": self.input,
            "date": self.date,
            "output": self.output,
            "context": self.context,
            "email": self.email
        }