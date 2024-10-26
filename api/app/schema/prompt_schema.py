from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class Prompt(BaseModel):
    # id: Optional[str]  # Use `str` type instead of `ObjectId`
    input: str
    # date: datetime
    # output: Optional[str]
    context: Optional[str]
    # status: Optional[str]

    class Config:
        from_attributes = True
        populate_by_name = True
