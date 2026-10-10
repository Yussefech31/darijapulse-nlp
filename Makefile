.PHONY: help setup data benchmark test lint format run-api run-ui docker-up docker-down clean

PYTHON ?= python

help:
	@echo "DarijaPulse NLP - Moroccan Darija Sentiment & Intent Platform"
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies in editable mode"
	@echo "  make data       - Fetch and cache real Moroccan Darija sentiment dataset"
	@echo "  make benchmark  - Run evaluation across TF-IDF, BiLSTM, and INT8 ONNX models"
	@echo "  make test       - Run pytest test suite"
	@echo "  make lint       - Run ruff linter checks"
	@echo "  make format     - Run ruff code formatting"
	@echo "  make run-api    - Start FastAPI REST backend on port 8000"
	@echo "  make run-ui     - Start Streamlit interactive analyzer on port 8501"
	@echo "  make docker-up  - Build and launch Docker containers"
	@echo "  make clean      - Clean cache and build artifacts"

setup:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/fetch_data.py

benchmark:
	$(PYTHON) scripts/benchmark_models.py

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

run-api:
	$(PYTHON) -m uvicorn src.darijapulse.api.app:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
