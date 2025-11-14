"""This file generates a report from evaluation CSV files."""

import os
import sys
from ast import literal_eval
from typing import Any

import pandas as pd

NEWLINE = "\n"
BAR = "================================================"
EVALS_PATH = "evals/output/"


def create_report_header(title: str, main_df: pd.DataFrame, failed_response_df: pd.DataFrame) -> str:
    """
    Create a header for the report.
    
    Parameters
    ----------
    title : str
        The title of the report.
    main_df : pd.DataFrame
        The DataFrame containing the main evaluation results. This is a joined DataFrame of the agent response and metadata DataFrames.
    failed_response_df : pd.DataFrame
        The DataFrame containing the failed response records.

    Returns
    -------
    str
        The header string.
    """
    header = ""
    header += f"{title}" + NEWLINE + NEWLINE
    header += f"Total Questions: {len(main_df)}" + NEWLINE
    header += f"Total Failed Questions: {len(failed_response_df)}" + NEWLINE 
    if not failed_response_df.empty:
        header += "Failed Question IDs:" + NEWLINE
        header += NEWLINE.join(failed_response_df["question_id"].tolist()) + NEWLINE + NEWLINE
    header += NEWLINE
    header += BAR
    header += NEWLINE
    return header

def create_report_str_from_row(row: dict[str, Any]) -> str:
    row_report = f"Question: {row['question']}" + NEWLINE
    row_report += f"Model: {row['model']}" + NEWLINE
    row_report += f"Available Tools: {row['available_tools']}" + NEWLINE
    row_report += f"Number of Messages: {row['num_messages']}" + NEWLINE
    row_report += f"Number of LLM Calls: {row['num_llm_calls']}" + NEWLINE
    row_report += f"Number of Tool Calls: {row['num_tool_calls']}" + NEWLINE
    row_report += (
        f"Called Tools: {', '.join([x['name'] + '-' + str(x['args']) for x in row['called_tools']])}"
        + NEWLINE
    )

    # Add any generated Cypher queries
    if row["generated_cypher"]:
        row_report += NEWLINE
        row_report += f"Text2Cypher Queries:\n{NEWLINE.join([f'{idx + 1}.)' + NEWLINE + cypher.get('query') for idx, cypher in enumerate(row['generated_cypher'])])}\n"
    
    # Add evaluation metrics
    row_report += NEWLINE
    row_report += "Evaluation Metrics:" + NEWLINE
    row_report += NEWLINE
    row_report += f"Rouge F1 Score: {row['rouge_f1_score']}" + NEWLINE
    row_report += f"Factual Correctness F1 Score: {row['factual_correctness_f1_score']}" + NEWLINE
    row_report += f"Answer Relevancy Score: {row['answer_relevancy_score']}" + NEWLINE
    row_report += NEWLINE

    # Add errors if they exist
    if not pd.isna(row["agent_response_error"]) or not pd.isna(row["metadata_error"]):
        row_report += f"Errors:" + NEWLINE
        if not pd.isna(row["agent_response_error"]):
            row_report += f"Agent Response Error: {row['agent_response_error']}" + NEWLINE
        if not pd.isna(row["metadata_error"]):
            row_report += f"Metadata Error: {row['metadata_error']}" + NEWLINE

    # Add the final answer
    row_report += NEWLINE
    row_report += f"Final Answer: {row['agent_final_answer']}"

    return row_report


def pretty_print(row: dict[str, Any]) -> None:
    print(create_report_str_from_row(row))
    print()
    print(BAR)
    print()


def _create_report_str(title: str, main_df: pd.DataFrame, failed_response_df: pd.DataFrame) -> str:
    report = create_report_header(title, main_df, failed_response_df)
    for idx, row in main_df.iterrows():
        report += f"{idx + 1}.)" + NEWLINE
        report += create_report_str_from_row(row)
        report += NEWLINE + NEWLINE
        report += BAR
        report += NEWLINE + NEWLINE
    return report


def create_report_from_dataframes(title: str, agent_response_df: pd.DataFrame, metadata_df: pd.DataFrame, failed_response_df: pd.DataFrame) -> str:
    """
    Generate a report from a collection of evaluation results DataFrames. 
    These DataFrames should be the unaltered output of the evaluation workflow.
    
    Parameters
    ----------
    title : str
        The title of the report.
    agent_response_df : pd.DataFrame
        The DataFrame containing the agent response records.
    metadata_df : pd.DataFrame
        The DataFrame containing the metadata records.
    failed_response_df : pd.DataFrame
        The DataFrame containing the failed response records.

    Returns
    -------
    str
        The report string.
    """

    main_df = pd.merge(agent_response_df[["question_id", 
                                          "rouge_f1_score", 
                                          "factual_correctness_f1_score", 
                                          "answer_relevancy_score", 
                                          "error"]].rename(columns={"error": "agent_response_error"}), 
                                          metadata_df.rename(columns={"error": "metadata_error"}), 
                                          on="question_id")
    
    return _create_report_str(title, main_df, failed_response_df)

def create_report_from_directory(title: str, directory_path: str) -> str:
    """
    Create a report from a directory of evaluation CSV files.

    Parameters
    ----------
    title : str
        The title of the report.
    directory_path : str
        The path to the directory containing the evaluation CSV files.

    Returns
    -------
    str
        The report string.
    """
    agent_response_df = pd.read_csv(os.path.join(directory_path, "agent_response.csv"))
    metadata_df = pd.read_csv(os.path.join(directory_path, "metadata.csv"), converters={
            "generated_cypher": literal_eval,
            "called_tools": literal_eval,
        })
    failed_response_df = pd.read_csv(os.path.join(directory_path, "failed_response.csv"))

    return create_report_from_dataframes(title, agent_response_df, metadata_df, failed_response_df)
    


def get_most_recent_eval_run_directory() -> str:
    """Get the most recent evaluation run directory."""
    eval_runs = [
        x for x in os.listdir(EVALS_PATH)
        if x.startswith("eval_run_") and os.path.isdir(os.path.join(EVALS_PATH, x))
    ]
    if not eval_runs:
        raise ValueError(f"No evaluation run directories found in {EVALS_PATH}")
    eval_runs.sort()
    return eval_runs[-1]


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 1:
        directory_path = get_most_recent_eval_run_directory()
    else:
        directory_path = args[0]

    title = f"Evaluation Report {directory_path.lstrip('eval_run_')}"
    report = create_report_from_directory(
        title=title,
        directory_path=EVALS_PATH + directory_path,
    )

    with open(f"{EVALS_PATH}{directory_path}/report.txt", "w") as f:
        f.write(report)
