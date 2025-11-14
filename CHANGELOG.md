## Next

### Fixed 

### Changed
* Update the eval workflow to intialize agent only once and use different thread ids for each question / conversation
* Each eval run has it's own directory in `evals/output/` containing `agent_response.csv`, `failed_response.csv` and `metadata.csv`
* Refactor evaluation workflow to update eval records in place

### Added
* Add RAGAS metrics to evaluation workflow - `rouge_f1_score`, `factual_correctness_f1_score`, `answer_relevancy_score`

## v0.1.0
* Single file agent using LangGraph
* Simple evaluation framework that captures metadata about the agent response
* Script to generate a report based on evaluation CSV result