from datetime import datetime
from pydantic import BaseModel
from typing import Optional


class Prompt(BaseModel):
    id: Optional[str]  # Use `str` type instead of `ObjectId`
    input: str
    date: datetime
    output: Optional[str]

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        fields = {'id': '_id'}
