"""This file generates a report from an evaluation CSV file."""

import os
import sys
from ast import literal_eval
from typing import Any

import pandas as pd

NEWLINE = "\n"
EVALS_PATH = "evals/output/"
REPORTS_PATH = "evals/reports/"


def create_report_str_from_row(row: dict[str, Any]) -> str:
    report = f"Question: {row['question']}" + NEWLINE
    report += f"Model: {row['model']}" + NEWLINE
    report += f"Available Tools: {row['available_tools']}" + NEWLINE
    report += f"Number of Messages: {row['num_messages']}" + NEWLINE
    report += f"Number of LLM Calls: {row['num_llm_calls']}" + NEWLINE
    report += f"Number of Tool Calls: {row['num_tool_calls']}" + NEWLINE
    report += (
        f"Called Tools: {', '.join([x['name'] + '-' + str(x['args']) for x in row['called_tools']])}"
        + NEWLINE
    )
    if row["generated_cypher"]:
        report += f"Text2Cypher Queries:\n{NEWLINE.join([f'{idx + 1}.)' + NEWLINE + cypher.get('query') for idx, cypher in enumerate(row['generated_cypher'])])}\n"
    if not pd.isna(row["error"]):
        report += f"Error: {row['error']}" + NEWLINE
    report += f"Final Answer: {row['agent_final_answer']}"
    return report


def pretty_print(row: dict[str, Any]) -> None:
    print(create_report_str_from_row(row))
    print()
    print("================================================")
    print()


def create_report(dataframe: pd.DataFrame) -> str:
    """Generate a report from an evaluation results DataFrame."""

    report = ""
    for idx, row in dataframe.iterrows():
        report += f"{idx + 1}.)" + NEWLINE
        report += create_report_str_from_row(row)
        report += NEWLINE + NEWLINE
        report += "================================================"
        report += NEWLINE + NEWLINE
    return report


def get_most_recent_eval_csv() -> str:
    """Get the most recent evaluation CSV file."""
    eval_csv = [x for x in os.listdir(EVALS_PATH) if x.endswith(".csv") and "benchmark" in x]
    eval_csv.sort()
    return eval_csv[-1]


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 1:
        csv_name = get_most_recent_eval_csv()
    else:
        csv_name = args[0]

    data = pd.read_csv(
        EVALS_PATH + csv_name,
        converters={
            "generated_cypher": literal_eval,
            "called_tools": literal_eval,
        },
    )

    with open(f"{REPORTS_PATH}{csv_name[:-4]}_report.txt", "w") as f:
        f.write(create_report(data))
