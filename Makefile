.PHONY: all format help

all: help

format:
	uv run ruff format .
	uv run ruff check --select I . --fix
	uv run ruff check .

run-agent-uv:
	uv run python3 agent.py

run-agent:
	python3 agent.py

help:
	@echo '----'
	@echo 'format...................... - run code formatters'
	@echo 'run-agent-uv................ - run agent using uv'
	@echo 'run-agent................... - run agent in current env'