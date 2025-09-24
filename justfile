install:
  poetry install
  poetry run pre-commit install

lint:
  poetry run pre-commit run --all-files
