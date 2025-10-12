install:
  poetry install

lint:
  poetry run pre-commit run --all-files

test: install
  poetry run pytest tests/unit

test-integration: install
  poetry run pytest tests/integration -m integration -n 0

test-all: install
  poetry run pytest .
