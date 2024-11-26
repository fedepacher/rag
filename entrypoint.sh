#!/bin/sh

# Start the Ollama server in the background
ollama serve &

# Wait for the Ollama server to be ready
echo "Waiting for Ollama server to start..."
until curl -s http://localhost:11434/api/ps > /dev/null; do
  echo "Ollama is not ready yet. Waiting..."
  sleep 2
done
echo "Ollama server is running."

# Check if the Mistral model is already downloaded
echo "Checking if the Mistral model is already downloaded..."
if curl -s http://localhost:11434/api/models | grep -q '"name": "mistral"'; then
  echo "Mistral model is already downloaded."
else
  echo "Mistral model not found. Downloading..."
  curl -X POST http://localhost:11434/api/pull -d '{
    "name": "mistral"
  }'
fi

# Start your Python script
echo "Starting the Python script..."
python rag/main.py

