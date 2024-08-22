import logging
from fastapi import HTTPException, status

from api.app.model.user_model import Users as UserModel
from api.app.schema import user_schema
from api.app.service.auth_service import get_password_hash


def create_user(user: user_schema.UserRegister):
    """Create users and store them in the DB.

    Args:
        user (user_schema.UserRegister): User parameters.

    Raises:
        HTTPException: If the user is already created.

    Returns:
        user_schema.User: User schema.
    """
    logging.info(f"Creating user: {user.username}")
    get_user = UserModel.filter((UserModel.email == user.email)
                                | (UserModel.username == user.username)).first()
    if get_user:
        msg = "Email already registered"
        if get_user.username == user.username:
            msg = "Username already registered"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg
        )

    db_user = UserModel(
        username=user.username,
        email=user.email,
        password=get_password_hash(user.password)
    )
    logging.info(f"Saving user: {user.username} in database")
    db_user.save()

    return user_schema.User(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email
    )
