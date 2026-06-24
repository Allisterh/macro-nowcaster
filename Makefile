.PHONY: install test lint build api app docker
install:; pip install -e ".[dev,app,llm]"
test:; pytest -q
lint:; ruff check src tests
build:; python -m macro_nowcaster.pipeline
api:; uvicorn macro_nowcaster.api.main:app --reload --port 8000
app:; MN_API_URL=http://localhost:8000 streamlit run app/streamlit_app.py
docker:; docker compose up --build
