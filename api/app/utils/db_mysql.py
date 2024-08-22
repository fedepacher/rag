import logging
import peewee
from contextvars import ContextVar
from fastapi import Depends

from api.app.utils.settings import Settings


settings = Settings()

DB_NAME = settings.db_name
DB_USER = settings.db_user
DB_PASS = settings.db_pass
DB_HOST = settings.db_host
DB_PORT = settings.db_port

assert DB_NAME is not None, "DB_NAME env var not set"
assert DB_USER is not None, "DB_USER env var not set"
assert DB_PASS is not None, "DB_PASS env var not set"
assert DB_HOST is not None, "DB_HOST env var not set"
assert DB_PORT is not None, "DB_PORT env var not set"

db_state_default = {'closed' : None, 'conn' : None, 'ctx' : None, 'transactions' : None}
db_state = ContextVar('db_state', default=db_state_default.copy())


class PeeweeConnectionState(peewee._ConnectionState):

    def __init__(self, **kwargs):
        super().__setattr__('_state', db_state)
        super().__init__(**kwargs)


    def __setattr__(self, name, value):
        """Set name attribute.

        Args:
            name (str): Name.
            value (str): Name Value.
        """
        self._state.get()[name] = value


    def __getattr__(self, name):
        """Get name attribute.

        Args:
            name (str): Name.
        """
        return self._state.get()[name]


# db = peewee.PostgresqlDatabase(DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
logging.info(f"Connecting to local database {DB_HOST}")
db = peewee.MySQLDatabase(DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=int(DB_PORT))
db._state = PeeweeConnectionState()


async def reset_db_state():
    """Reset DB state."""
    db._state._state.set(db_state_default.copy())
    db._state.reset()


def get_db(db_state=Depends(reset_db_state)):
    """DB connection.

    Args:
        db_state (_type_, optional): DB state. Defaults to Depends(reset_db_state).
    """
    try:
        db.connect()
        yield
    finally:
        if not db.is_closed():
            db.close()
