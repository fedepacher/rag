from api.app.model.user_model import Users
from api.app.utils.db_mysql import db


def create_tables():
    """Create DB tables."""
    with db:
        db.create_tables([Users], safe=True)


if __name__ == '__main__':
    # Entry point for the one-shot schema step the container runs before gunicorn.
    # It must not run from an imported module: gunicorn forks several workers, and
    # concurrent DDL against an empty schema raced on the index creation. peewee's
    # safe=True cannot help there -- MySQL has no CREATE INDEX IF NOT EXISTS, so
    # MySQLDatabase.safe_create_index is False and peewee's early return only covers a
    # run where the table already exists.
    create_tables()
