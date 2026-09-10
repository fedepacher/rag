import logging
from fastapi import HTTPException, status
from pymongo import ASCENDING

from api.app.schema import prompt_schema, user_schema
from api.app.model.prompt_model import Prompt as PromptModel
from api.app.utils.db_nosql import mongo_collection  # Import the MongoDB collection
from api.app.utils.serialization import serialize_mongo_document


def input_prompt(prompt: prompt_schema.Prompt, user: user_schema.User):
    """Create a new entrance in the database.

    Args:
        prompt (prompt_schema.Prompt): Entrance to add to the DB.
        user (user_schema.Use): User registered.

    Returns:
        weather_schema.Weather: Entrance fields schema.
    """
    mongo_prompt = PromptModel(
        input=prompt.input,
        email=user.email
    )

    try:
        mongo_collection.insert_one(mongo_prompt.to_dict())
    except Exception as e:
        logging.error(f"Error loading prompt in database: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An error occurred while loading prompt in database.")

    return {
        "message": "Prompt successfully saved in MongoDB"
    }


async def get_prompts():
    """Get all the prompts in the db that contains output NULL

    Returns:
        list: List of all the prompt without answer.
    """
    # Query for documents where output is None and sort by date_in
    try:
        document = mongo_collection.find_one({'output': None}, sort=[("date_in", ASCENDING)])
    except Exception:
        logging.exception("Failed to read the prompt queue from MongoDB")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An error occurred while retrieving prompts.")

    # An empty queue is the normal idle state, not an error. Returning None lets the
    # router answer 204, which is the branch rag/message_clients.py already implements
    # (an INFO log and EMPTY_QUEUE_INTERVAL_SEC). Previously the empty case raised a 404
    # from INSIDE the try above, the bare `except Exception` caught it and re-raised it
    # as a 500, and the RAG service logged "Received response code 500" every 15 seconds
    # while nothing at all was wrong.
    if document is None:
        return None

    return serialize_mongo_document(document)
