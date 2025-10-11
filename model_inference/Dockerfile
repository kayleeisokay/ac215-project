FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./

# Install PyTorch CPU first (smaller download)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Then install the rest with UV
RUN pip install uv
RUN uv pip install --system -r pyproject.toml

COPY . .
ENV GOOGLE_APPLICATION_CREDENTIALS=/credentials/dosewise-473716-9f4874e812d6.json
EXPOSE 8000
CMD ["python", "inference.py"]


