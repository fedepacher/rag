import logging
from fastapi import HTTPException, status

from api.app.schema import prompt_schema
from api.app.model.prompt_model import Prompt as PromptModel
from api.app.utils.db_nosql import mongo_collection  # Import the MongoDB collection


def input_prompt(prompt: prompt_schema.Prompt):
    """Create a new entrance in the database.

    Args:
        prompt (prompt_schema.Prompt): Entrance to add to the DB.

    Returns:
        weather_schema.Weather: Entrance fields schema.
    """
    mongo_prompt = PromptModel(
        input=prompt.input
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
