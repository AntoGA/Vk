.PHONY: install dev test lint format clean docker-build docker-up docker-down

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest tests/ -v --cov=ml --cov=api --cov-report=html

lint:
	flake8 ml api dashboard
	mypy ml api

format:
	black ml api dashboard tests
	isort ml api dashboard tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

train:
	python scripts/train_model.py

collect-data:
	python scripts/collect_data.py

benchmark:
	python scripts/benchmark.py
