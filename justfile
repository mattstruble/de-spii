install:
  poetry install

lint:
  poetry run pre-commit run --all-files

test: install
  poetry run pytest .
