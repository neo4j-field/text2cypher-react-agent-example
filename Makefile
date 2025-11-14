.PHONY: all format help

all: help

format:
	uv run ruff format .
	uv run ruff check --select I . --fix
	uv run ruff check .

export-requirements:
	uv export --no-hashes > requirements.txt

run-agent-uv:
	uv run python3 single_file_agent.py

run-agent:
	python3 single_file_agent.py

run-eval-uv:
	uv run python3 -m evaluation.main

run-eval:
	python3 -m evaluation.main

generate-report-uv:
	uv run python3 scripts/generate_report.py $(csv-name)

generate-report:
	python3 scripts/generate_report.py $(csv-name)

help:
	@echo '----'
	@echo 'Available commands:'
	@echo ''
	@echo 'Agent Commands:'
	@echo '  run-agent-uv.............. - run agent using uv'
	@echo '  run-agent................. - run agent in current env'
	@echo ''
	@echo 'Evaluation Commands:'
	@echo '  run-eval-uv............... - run evaluation suite using uv'
	@echo '  run-eval.................. - run evaluation suite in current env'
	@echo '  generate-report-uv........ - generate report from eval results using uv (requires csv-name=<file>)'
	@echo '  generate-report........... - generate report from eval results in current env (requires csv-name=<file>)'
	@echo ''
	@echo 'Development Commands:'
	@echo '  format.................... - run code formatters (ruff)'
	@echo '  export-requirements....... - export dependencies from uv.lock to requirements.txt'
	@echo ''
	@echo 'Examples:'
	@echo '  make run-agent-uv'
	@echo '  make run-eval-uv'
	@echo '  make generate-report-uv csv-name=eval_benchmark_results_2025-11-13_08-14-28'
	@echo '  make export-requirements'
	@echo '----'