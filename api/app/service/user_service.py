import json
import logging
from fastapi import HTTPException, status
from peewee import OperationalError

from api.app.model.user_model import Users as UserModel
from api.app.schema import user_schema
from api.app.service.auth_service import get_password_hash, get_user, verify_password
from api.app.utils.email_handler import EmailSender
from api.app.utils.settings import Settings
from api.app.utils.password_generator import generate_password


settings = Settings()
EMAIL_SERVER = settings.email_server
EMAIL = settings.email
with open('passwords.json', 'r') as file:
    data = json.load(file)
EMAIL_PASSWORD = data.get('forgot_password')
EMAIL_PROTOCOL = settings.email_protocol
EMAIL_BODY = settings.email_body
EMAIL_SUBJECT = settings.email_subject

email_sender = EmailSender(server=EMAIL_SERVER, email=EMAIL , password=EMAIL_PASSWORD, security_protocol=EMAIL_PROTOCOL)


def create_user(user: user_schema.UserRegistered):
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


async def forgot_password(username_or_email: str, language: str):
    logging.info(f"Checking user {username_or_email} in database to send email with new password")
    user = get_user(username_or_email)
    if not user:
        logging.debug(f"User {username_or_email} does not exist")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username",
            headers={"WWW-Authenticate": "Bearer"}
        )
    email_file_body = EMAIL_BODY.replace('.txt', f'_{language.lower()}.txt')
    email_file_subject = EMAIL_SUBJECT.replace('.txt', f'_{language.lower()}.txt')

    new_password = generate_password()
    # Hash the new password
    hashed_password = get_password_hash(new_password)
    # Get email content
    with open(email_file_subject, 'r', encoding='utf-8') as file:
        email_subject = file.read()
    with open(email_file_body, 'r', encoding='utf-8') as file:
        email_content = file.read()

    # Replace placeholders with actual values
    email_content = email_content.replace('{{username}}', user.username)
    email_content = email_content.replace('{{new_password}}', new_password)

    # Update the user's password in the database
    try:
        user.password = hashed_password
        user.save()
        logging.info(f"Password for user {user.username} updated successfully in the database")
    except OperationalError as e:
        logging.error(f"Database error while updating password: {e}")
        raise HTTPException(status_code=500, detail="Failed to update password")

    await email_sender.send(email_subject, email_content, user.email)

    return {"message": "Password reset email sent"}


def change_password(password_data: user_schema.UserPasswordHandler, user: user_schema.User):
    # Get user from db to get the password hash
    logging.info(f"Getting user {user.username} from database")
    user_db = get_user(user.username)
    # Get stored password
    stored_password_hash = user_db.password

    if not verify_password(password_data.password, stored_password_hash):
        logging.info("Stored password does not match the entered old password")
        raise HTTPException(status_code=401, detail="Old password is incorrect")

        # Check if new password matches the re-written password
    if password_data.new_password != password_data.rewrite_password:
        logging.info("New password does not match the re-written password")
        raise HTTPException(status_code=400, detail="New password and re-written password do not match")

    hashed_password = get_password_hash(password_data.new_password)

    try:
        user.password = hashed_password
        user.save()
        logging.info(f"Password for user {user.username} updated successfully in the database")
    except OperationalError as e:
        logging.error(f"Database error while updating password: {e}")
        raise HTTPException(status_code=500, detail="Failed to update password")

    return {"message": "Password changed successfully"}
