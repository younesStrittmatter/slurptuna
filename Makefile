.VENV := .venv/bin
.PHONY: docs docs-serve docs-build

docs-build:
	$(.VENV)/python scripts/build_examples_docs.py
	$(.VENV)/python scripts/build_api_reference.py
	$(.VENV)/mkdocs build

docs-serve:
	$(.VENV)/python scripts/build_examples_docs.py
	$(.VENV)/python scripts/build_api_reference.py
	$(.VENV)/mkdocs serve

docs: docs-build

