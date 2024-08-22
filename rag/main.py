import os
import logging
import time
from pymongo import MongoClient
from bson import ObjectId

from rag.message_clients import APIClient, MessageClient


RESTART_AFTER_EXCEPTION_DELAY_SEC = 60


class MessageProcessor:
    def __init__(self, api: MessageClient, mongo_collection):
        self.api = api
        self.mongo_collection = mongo_collection

    def run(self):
        logging.info("Start processing messages")
        try:
            for i, message in enumerate(self.api.messages()):
                logging.info(f"{message.input}: Processing prompt...")

                # Process the input and generate the output
                output = 'respuesta' # self.process_input(message_data.input)

                # Update the MongoDB document with the new output value
                self.mongo_collection.update_one(
                    {"_id": ObjectId(message.id)},
                    {"$set": {"output": output}}
                )
                logging.info(f"Message {message.id} updated with output: {output}")


        except Exception as e:
            logging.error(f"Error processing message: {e}")

    def run_continuous(self):
        while True:
            self.run()
            time.sleep(RESTART_AFTER_EXCEPTION_DELAY_SEC)


def main(api_url, document_location, mongo_host, mongo_port, mongo_user, mongo_pass, mongo_db_name):
    logging.info("Starting system")
    api_client = APIClient(api_url, token=None)
    logging.info(f"Connecting to local database {MONGO_HOST}")
    client = MongoClient(f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/")
    mongo_db = client[MONGO_DB_NAME]

    mongo_collection = mongo_db['prompts']  # Collection name
    message_processor = MessageProcessor(api_client, mongo_collection)
    message_processor.run_continuous()


if __name__ == '__main__':
    API_URL = os.environ.get('API_URL', None)
    DOCUMENT_LOCATION = os.getenv("DOCUMENT_LOCATION", None)
    MONGO_HOST = os.getenv('MONGO_HOST')
    MONGO_PORT = (os.getenv('MONGO_PORT'))
    MONGO_USER = os.getenv('MONGO_USER')
    MONGO_PASS = os.getenv('MONGO_PASS')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')  # Replace with your MongoDB database name

    main(API_URL, DOCUMENT_LOCATION, MONGO_HOST, MONGO_PORT, MONGO_USER, MONGO_PASS, MONGO_DB_NAME)
