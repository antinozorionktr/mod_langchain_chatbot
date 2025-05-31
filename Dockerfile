# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directory for vector database
RUN mkdir -p ./vector_db

# Expose ports
EXPOSE 8501 11434

# Create startup script
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV OLLAMA_HOST=0.0.0.0:11434

# Run the startup script
CMD ["/startup.sh"]