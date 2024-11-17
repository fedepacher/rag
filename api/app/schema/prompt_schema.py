from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class Prompt(BaseModel):
    input: str

    class Config:
        from_attributes = True
        populate_by_name = True
