#!/bin/sh

# Start the Ollama server in the background
ollama serve &

# Wait for the Ollama server to be ready (timeout after 30 seconds)
echo "Waiting for Ollama server to start..."
TIMEOUT=30
ELAPSED=0
until curl -s http://localhost:11434/api/ps > /dev/null; do
  echo "Ollama is not ready yet. Waiting..."
  sleep 2
  ELAPSED=$((ELAPSED + 2))
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "Error: Ollama server failed to start within $TIMEOUT seconds."
    exit 1
  fi
done
echo "Ollama server is running."

# Check if the Mixtral model is already downloaded
MODEL_NAME="mixtral:8x7b"
echo "Checking if the $MODEL_NAME model is already downloaded..."
if curl -s http://localhost:11434/api/tags | grep -q "\"name\": \"$MODEL_NAME\""; then
  echo "$MODEL_NAME model is already downloaded."
else
  echo "$MODEL_NAME model not found. Downloading..."
  # Pull the model and check for success
  RESPONSE=$(curl -s -X POST http://localhost:11434/api/pull -d "{\"name\": \"$MODEL_NAME\"}")
  if echo "$RESPONSE" | grep -q '"status": "success"'; then
    echo "$MODEL_NAME model downloaded successfully."
  else
    echo "Error: Failed to download $MODEL_NAME model."
    echo "Response: $RESPONSE"
    exit 1
  fi
fi

# Verify the model is available
echo "Verifying $MODEL_NAME model availability..."
if ! curl -s http://localhost:11434/api/tags | grep -q "\"name\": \"$MODEL_NAME\""; then
  echo "Error: $MODEL_NAME model not found after pull attempt."
  exit 1
fi

# Start the Python script
echo "Starting the Python script..."
python /app/rag/main.py
if [ $? -ne 0 ]; then
  echo "Error: Python script failed to execute."
  exit 1
fi