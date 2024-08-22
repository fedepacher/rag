from datetime import datetime
from pydantic import BaseModel, Field


class Prompt(BaseModel):
    id: int
    input: str = Field(..., max_length=500)
    date: datetime = Field(default_factory=datetime.now)
    output: str = Field(..., max_length=500)

    class Config:
        from_attributes = True
