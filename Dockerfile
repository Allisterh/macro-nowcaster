FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY config ./config
RUN pip install --upgrade pip && pip install -e .

EXPOSE 8000
# Build the artifact once at startup, then serve the API.
CMD ["sh", "-c", "python -m macro_nowcaster.pipeline && uvicorn macro_nowcaster.api.main:app --host 0.0.0.0 --port 8000"]
