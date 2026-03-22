.VENV := .venv/bin
.PHONY: docs docs-serve docs-build

docs-build:
	python scripts/build_examples_docs.py
	python scripts/build_api_reference.py
	$(.VENV)/mkdocs build

docs-serve:
	python scripts/build_examples_docs.py
	python scripts/build_api_reference.py
	$(.VENV)/mkdocs serve

docs: docs-build

