from api.app.model.user_model import Users
from api.app.utils.db_mysql import db


def create_tables():
    """Create DB tables."""
    with db:
        db.create_tables([Users], safe=True)
