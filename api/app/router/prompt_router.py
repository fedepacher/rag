from datetime import datetime
from fastapi import APIRouter, Depends, Body, status, Query, Path
from typing import List, Optional

from api.app.schema import prompt_schema
from api.app.service import prompt_service
from api.app.utils.db_mysql import get_db
from api.app.schema.user_schema import User
from api.app.service.auth_service import get_current_user


router = APIRouter(prefix="/prompt")

@router.post(
    "/",
    tags=["prompt"],
    status_code=status.HTTP_201_CREATED,
    # response_model=prompt_schema.Prompt,
    dependencies=[Depends(get_db)]
)
def input_prompt(prompt: prompt_schema.Prompt=Body(...),
                 current_user: User =Depends(get_current_user)):

    return prompt_service.input_prompt(prompt)
