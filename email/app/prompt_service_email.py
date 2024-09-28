#desde api/app/ archivo de nombre semejante (lo modificaré según convenga)
import logging
from fastapi import HTTPException, status
from pymongo import ASCENDING

from db_nosql import mongo_collection  # Import the MongoDB collection

from bson.objectid import ObjectId 

# from api.app.schema import prompt_schema, user_schema
# from api.app.model.prompt_model import Prompt as PromptModel
from prompt_model import Prompt as PromptModel
# from api.app.utils.serialization import serialize_mongo_document


def input_prompt(prompt: PromptModel):
    """Create a new entrance in the database.

    Args:
        prompt (PromptModel): Entrance to add to the DB.

    Returns:
        None
    """

    try:
        mongo_collection.insert_one(prompt.to_dict())
    except Exception as e:
        logging.error(f"Error loading prompt in database: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An error occurred while loading prompt in database.")




def get_one_to_send():
    """Get a in the db that contains output NOT NULL and status NULL

    Returns:
        dict: A document with the answer if found, otherwise None.
    """
    try:
        document = mongo_collection.find_one({'output': {'$ne': None}, 'status': None})     #, sort=[("date", ASCENDING)])
        return document

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An error occurred while retrieving prompts.")



def update_status_doc(document_id: str, new_status: str):
    """Update the status of a document based on its ID.

    Args:
        document_id (str): The ID of the document to update.
        new_status (str): The new status to set.

    Returns:
        None.
    """
    try:
        # Asegura ObjectId format
        if not ObjectId.is_valid(document_id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document ID format")

        object_id = ObjectId(document_id)

        # Perform the update, setting the 'status' field to the new value
        result = mongo_collection.update_one({'_id': object_id}, {'$set': {'status': new_status}})

        if result.matched_count == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An error occurred while updating the document: {e}")


def update_doc(document_id: str, new_status: str):
    """Update the status of a document based on its ID.

    Args:
        document_id (str): The ID of the document to update.
        new_status (str): The new status to set.

    Returns:
        None
    """
    try:
        object_id = ObjectId(document_id)
        mongo_collection.update_one({'_id': object_id}, {'$set': {'status': new_status}})

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An error occurred while updating the document: {e}")