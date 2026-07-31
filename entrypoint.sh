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

# Generation model (llama3.1) plus the CRAG control model (phi3.5), which grades
# retrieval relevance and reformulates queries. Both are loaded at runtime.
MODELS="llama3.1:8b-instruct-q4_K_M phi3.5:3.8b-mini-instruct-q4_K_M"

for model in $MODELS; do
  echo "Checking if the $model model is already downloaded..."
  if ollama list | grep -qF "$model"; then
    echo "$model model is already downloaded."
  else
    echo "$model model not found. Downloading..."
    ollama pull "$model"
  fi
done

# Start your Python script
echo "Starting the Python script..."
python rag/main.py