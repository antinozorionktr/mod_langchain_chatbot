#!/bin/bash

# Start Ollama service in background
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done

echo "Ollama is ready!"

# Pull required models
echo "Pulling Llama 3.1 model..."
ollama pull llama3.1

echo "Pulling embedding model..."
ollama pull nomic-embed-text

echo "Models downloaded successfully!"

# Start Streamlit app
echo "Starting Streamlit application..."
streamlit run main.py --server.port=8501 --server.address=0.0.0.0