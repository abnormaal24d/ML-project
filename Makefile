.PHONY: typecheck

typecheck:
	python -m mypy . --no-incremental --config-file pyproject.toml
