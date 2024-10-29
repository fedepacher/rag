from pydantic import BaseModel
from pydantic import Field
from pydantic import EmailStr


class UserBase(BaseModel):
    """User base class.

    Args:
        BaseModel (_type_): BaseModel checker.
    """
    email : EmailStr = Field(
        ...,
        example='myemail@mail.com'
    )
    username : str = Field(
        ...,
        min_length=3,
        max_length=50,
        example='myusername'
    )


class User(UserBase):
    """User class.

    Args:
        UserBase (_type_): BaseModel checker.
    """
    id : int = Field(
        ...,
        example='5'
    )


class UserSensitiveInformation(BaseModel):
    """Password registration class.

    Args:
        UserBase (_type_): BaseModel checker.
    """
    password : str = Field(
        ...,
        min_length=8,
        max_length=64,
        example='mypassword'
    )


class UserRegistered(UserBase, UserSensitiveInformation):
    pass


class UserPasswordHandler(UserSensitiveInformation):
    new_password: str = Field(..., min_length=8, max_length=64, example='newpassword')
    rewrite_password: str = Field(..., min_length=8, max_length=64, example='rewritenewpassword')

