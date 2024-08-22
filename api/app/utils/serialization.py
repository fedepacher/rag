from bson import ObjectId


def serialize_mongo_document(doc):
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    return doc
