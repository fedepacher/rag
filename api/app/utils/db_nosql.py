import logging

from pymongo import MongoClient


from api.app.utils.settings import Settings


settings = Settings()

MONGO_HOST = settings.mongo_host
MONGO_USER = settings.mongo_user
MONGO_PASS = settings.mongo_pass
MONGO_DB_NAME = settings.mongo_db_name
MONGO_PORT = settings.mongo_port

logging.info(f"Connecting to local database {MONGO_HOST}")
client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/")
mongo_db = client[MONGO_DB_NAME]

mongo_collection = mongo_db['prompts']  # Collection name