.PHONY: help install dev-install test test-unit test-integration lint format clean run evaluate

help:
	@echo Available commands:
	@echo   make install      - Install production dependencies
	@echo   make dev-install  - Install dev dependencies
	@echo   make test         - Run all tests with coverage
	@echo   make test-unit    - Run only unit tests (fast)
	@echo   make test-integration - Run integration tests (calls external APIs)
	@echo   make lint         - Run linting checks
	@echo   make format       - Format code
	@echo   make clean        - Clean cache and build files
	@echo   make run          - Run the API server
	@echo   make evaluate     - Run evaluation on golden dataset

install:
	uv pip install -e .

dev-install:
	uv pip install -e ".[dev]"

test:
	pytest

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration

test-verbose:
	pytest -v -s

lint:
	@echo Running ruff...
	ruff check .
	@echo Running mypy...
	mypy src/
	@echo Running black check...
	black --check .

format:
	@echo Running isort...
	isort .
	@echo Running black...
	black .
	@echo Running ruff fix...
	ruff check --fix .

clean:
	@echo Cleaning Python cache files...
	@if exist __pycache__ rd /s /q __pycache__
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (.ruff_cache) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (.mypy_cache) do @if exist "%%d" rd /s /q "%%d"
	@for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
	@if exist htmlcov rd /s /q htmlcov
	@if exist dist rd /s /q dist
	@if exist build rd /s /q build
	@if exist .coverage del .coverage
	@echo Done!

run:
	uvicorn reit_risk_summarizer.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn reit_risk_summarizer.main:app --host 0.0.0.0 --port 8000 --workers 4

evaluate:
	python -m evaluation.run_evaluation
