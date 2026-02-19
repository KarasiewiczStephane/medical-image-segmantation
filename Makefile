.PHONY: install test lint clean run dashboard docker docker-compose

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short --cov=src

lint:
	ruff check src/ tests/
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	python -m src.main

dashboard:
	streamlit run src/dashboard/app.py --server.port=8501

docker:
	docker build -t $(shell basename $(CURDIR)) .
	docker run -p 8501:8501 $(shell basename $(CURDIR))

docker-compose:
	docker compose up --build
