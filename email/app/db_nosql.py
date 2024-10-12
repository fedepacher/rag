#desde api/app/ archivo de igual nombre (comento las lineas que cambio)
from logger_config import logger

from pymongo import MongoClient


#from api.app.utils.settings import Settings
from config import Config

#settings = Settings()
settings = Config()

MONGO_HOST = settings.mongo_host
MONGO_USER = settings.mongo_user
MONGO_PASS = settings.mongo_pass
MONGO_DB_NAME = settings.mongo_db_name
MONGO_PORT = settings.mongo_port

logger.info(f"Connecting to local database {MONGO_HOST}")
client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/")
mongo_db = client[MONGO_DB_NAME]

mongo_collection = mongo_db['prompts']  # Collection name
