from datetime import datetime
from typing import Optional


class Prompt:
    def __init__(self, input: str, date: Optional[datetime] = None, output: Optional[str] = None):
        self.input = input
        self.date = date or datetime.now()
        self.output = output

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB insertion."""
        return {
            "input": self.input,
            "date": self.date,
            "output": self.output
        }