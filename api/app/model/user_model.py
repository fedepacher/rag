import peewee

from api.app.utils.db_mysql import db


class Users(peewee.Model):
    """DB user columns definition.

    Args:
        peewee (_type_): Validation model.
    """
    email = peewee.CharField(unique=True, index=True)
    username = peewee.CharField(unique=True, index=True)
    password = peewee.CharField(unique=True, index=True)


    class Meta:
        """DB connection"""
        database = db
