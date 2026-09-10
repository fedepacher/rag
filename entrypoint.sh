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

# Generation model (llama3.1), the CRAG control model (phi3.5) which grades retrieval
# relevance and reformulates queries, and the embedding model (bge-m3) behind the FAISS
# index. All three are served by this container's Ollama instance at runtime.
#
# Keep this list in step with OLLAMA_MODEL, PHI_MODEL and EMBEDDING_MODEL in rag/main.py.
# A model missing here is not a startup error: Ollama pulls it on first use instead, so
# the failure shows up as one inexplicably slow question rather than as a broken boot.
MODELS="llama3.1:8b-instruct-q4_K_M phi3.5:3.8b-mini-instruct-q4_K_M bge-m3"

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