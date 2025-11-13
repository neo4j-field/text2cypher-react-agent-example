.PHONY: all format help

all: help

format:
	uv run ruff format .
	uv run ruff check --select I . --fix
	uv run ruff check .

run-agent-uv:
	uv run python3 single_file_agent.py

run-agent:
	python3 single_file_agent.py

run-eval-uv:
	uv run python3 eval.py

run-eval:
	python3 eval.py

generate-report-uv:
	uv run python3 scripts/generate_report.py $(csv-name)

generate-report:
	python3 scripts/generate_report.py $(csv-name)

help:
	@echo '----'
	@echo 'format...................... - run code formatters'
	@echo 'run-agent-uv................ - run agent using uv'
	@echo 'run-agent................... - run agent in current env'