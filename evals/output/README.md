# Evaluation Outputs

This folder contains the output files from the evaluation process.

Depending on your package manager, you may run:
```
make run-eval-uv
# or 
make run-eval
```

Each evaluation run will create a new directory in `evals/output/` that contains the following:
* `agent_response.csv` - Contains evaluation metrics for the agent response. This includes metrics calculated with the RAGAS evaluation package.
* `failed_response.csv` - Contains any failed agent responses. These are failures where the agent could not generate a response for any reason.
* `metadata.csv` - Contains metadata about the agent response. This includes tool calls, response time and other information.
* `report.txt` - A report generated from the above csv files.