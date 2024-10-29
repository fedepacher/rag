import logging
from fastapi import APIRouter, Form, Depends, status, Body
from fastapi.security import OAuth2PasswordRequestForm

from api.app.schema import user_schema
from api.app.service import user_service
from api.app.service import auth_service
from api.app.schema.token_schema import Token
from api.app.utils.db_mysql import get_db
from api.app.service.auth_service import get_current_user


router = APIRouter(
    # prefix="/",
    tags=["users"]
)


@router.get("/")
def get_smoke():
    return {"Status": "API is up!"}


@router.post(
    "/user/",
    status_code=status.HTTP_201_CREATED,
    response_model=user_schema.User,
    dependencies=[Depends(get_db)],
    summary="Create a new user"
)
def create_user(user: user_schema.UserRegistered= Body(...)):
    """Create a new user in the app.

    Args:
        user (user_schema.UserRegister, optional): User params description. Defaults to Body(...).

    Returns:
        user: User info.
    """
    logging.info(f"Creating a new user in the Database")
    return user_service.create_user(user)


@router.post(
    "/login",
    tags=["users"],
    response_model=Token
)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login for access token.

    Args:
        form_data (OAuth2PasswordRequestForm, optional): Username/email or password. 
        Defaults to Depends().

    Returns:
        Token: Access token and token type.
    """
    logging.info(f"Login user")
    access_token = auth_service.generate_token(form_data.username, form_data.password)
    token = Token(access_token=access_token, token_type="bearer")
    return token


@router.post(
    "/auth/forgot-password/",
    status_code=status.HTTP_200_OK,
    summary="Send email with the new password"
)
async def forgot_password(username_or_email: str = Form(...), language: str = Form('es')):
    response = await user_service.forgot_password(username_or_email, language)
    return response


@router.post(
    "/auth/change-password/",
    status_code=status.HTTP_200_OK,
    summary="Change password"
)
def change_password(new_password: user_schema.UserPasswordHandler,
                    current_user: user_schema.User = Depends(get_current_user)):
    response = user_service.change_password(new_password, current_user)
    return response
