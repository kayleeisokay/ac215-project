FROM python:3.11-slim

# Install uv
RUN pip install uv

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Install dependencies using uv
RUN uv sync

# Copy application code
COPY . .

# Set environment variable (you'll mount the JSON file externally)
ENV GOOGLE_APPLICATION_CREDENTIALS=/credentials/dosewise-473716-9f4874e812d6.json

CMD ["uv", "run", "python", "app.py"]