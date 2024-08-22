from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from api.app.model.user_model import Users as UserModel
from api.app.schema.token_schema import TokenData
from api.app.utils.settings import Settings


settings = Settings()


SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.token_expire


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def verify_password(plain_password, password):
    """Validate password.

    Args:
        plain_password (str): Password stored in DB.
        password (str): Password.

    Returns:
        Bool: Password validation status.
    """
    return pwd_context.verify(plain_password, password)


def get_password_hash(password):
    """Generate hash.

    Args:
        password (str): Password.

    Returns:
        str: Password hashed.
    """
    return pwd_context.hash(password)


def get_user(username: str):
    """Get user or email depends if logging was with email or username.

    Args:
        username (str): Username or email.

    Returns:
        Any: User if exist.
    """
    return UserModel.filter((UserModel.email == username)
                            | (UserModel.username == username)).first()


def authenticate_user(username: str, password: str):
    """Check if username and password exists.

    Args:
        username (str): Username.
        password (str): Password

    Returns:
        str: Username
    """
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta]=None):
    """Create an access token.

    Args:
        data (dict): Data to store in token.
        expires_delta (Optional[timedelta], optional): Token expiration time. Defaults to None.

    Returns:
        str: Token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encode_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encode_jwt


def generate_token(username, password):
    """Generate access token if username and password exist.

    Args:
        username (str): Username.
        password (str): Password.

    Raises:
        HTTPException: HTTP 401 Exception.

    Returns:
        str: Token.
    """
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )


async def get_current_user(token: str=Depends(oauth2_scheme)):
    """Get the user information based on the token information. 

    Args:
        token (str, optional): Token. Defaults to Depends(oauth2_scheme).

    Raises:
        credentials_exception: Credential fail.
        credentials_exception: Credential fail.
        credentials_exception: Credential fail.

    Returns:
        _type_: _description_
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
