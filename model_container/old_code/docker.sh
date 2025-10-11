#!/bin/bash

# Get the absolute path to the project root (parent of model_container)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_PATH="${PROJECT_ROOT}/secrets/dosewise-473716-9f4874e812d6.json"

# Check if secrets file exists
if [ ! -f "$SECRETS_PATH" ]; then
    echo "Error: Secrets file not found at: $SECRETS_PATH"
    echo "Please follow the setup instructions in README.md"
    exit 1
fi

echo "Building Docker image..."
docker build -t baseline-model -f Dockerfile .

echo ""
echo "Running container and pulling data..."
docker run \
    --rm \
    --name baseline-model-container \
    -v "${SECRETS_PATH}:/app/dosewise-473716-9f4874e812d6.json" \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/dosewise-473716-9f4874e812d6.json \
    baseline-model -lc "source /home/app/.venv/bin/activate && python main.py"