# RAG System


## Docker deployment

docker compose up --build

## Get into the containers

docker exec -it rag-mongodb-1 bash

mongosh --host localhost --port 27017 -u mongoadmin -p secret --authenticationDatabase admin

use test_nosql

db.prompts.find().toArray()

db.prompts.find({ output: null }).toArray()

db.prompts.updateOne( { _id: ObjectId('67204a4ffbd470bb9c202696') }, { $set: { status: null } })

### Ordered by date
db.prompts.find({ output: null }).sort({ date: 1 }).toArray()

### Update

db.prompts.updateOne(
    { _id: ObjectId('66c9569e7fb497bf29917165') },
    { $set: { input: "En un transistor de efecto de campo de puerta aislada, el terminal de puerta maneja corriente?" } }
)


# api

DB_NAME: test_sql
DB_USER: rag-system
DB_PASS: rag-system
DB_HOST: localhost
DB_PORT: 3306
ACCESS_TOKEN_EXPIRE_MINUTES: 1440
SECRET_KEY: 6b3d75e968f26f3be3443503efefd761e22d5e173fea95646a3659e673ebb97b
MONGO_HOST: localhost
MONGO_PORT: 27017
MONGO_USER: mongoadmin
MONGO_PASS: secret
MONGO_DB_NAME: test_nosql
      
      

# rag
      
API_URL=http://127.0.0.1:8000
DOCUMENT_LOCATION=resources/files
MONGO_DB_NAME=test_nosql
MONGO_DB_NAME=test_nosql
MONGO_HOST=localhost
MONGO_PASS=secret
MONGO_PORT=27017
MONGO_USER=mongoadmin
PYTHONUNBUFFERED=1

# Ollama server

ollama serve

ollama run mistral