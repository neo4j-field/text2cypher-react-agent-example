import asyncio
import os
from datetime import datetime
from math import ceil
from time import perf_counter
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from evaluation.components.agent import create_agent_response_record
from evaluation.components.fail import create_failed_response_record
from evaluation.components.metadata import create_metadata_record
from evaluation.models import AgentResponseTableRecord, FailedResponseTableRecord, MetadataTableRecord, QuestionRecord
from prompt import get_movies_system_prompt
from tools.find_movie_recommendations import find_movie_recommendations_tool
from utils import get_questions_from_yaml, pre_model_hook

if load_dotenv():
    print("Loaded .env file")
else:
    print("No .env file found")

neo4j_cypher_mcp = StdioServerParameters(
    command="uvx",
    args=["mcp-neo4j-cypher@0.3.0", "--transport", "stdio"],
    env={
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
        "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),
    },
)

evals_loc = "evals/output/"
eval_results = list()


async def evaluate_single_question(
    agent: CompiledStateGraph,
    question_dict: dict[str, str],
    tools: list[StructuredTool],
    metadata_records: list[MetadataTableRecord],
    agent_response_records: list[AgentResponseTableRecord],
    failed_response_records: list[FailedResponseTableRecord],
    model: str = "openai:gpt-4.1",
) -> None:
    """
    Evaluate a single question in a new conversation thread.
    """

    try:
        assert question_dict.get("question") is not None, "Question not found"

        # create the thread id for the agent eval
        # use the question id if it exists, otherwise generate a random uuid
        # each question must have it's own thread id, so that we maintain 1 unique conversation thread per question
        thread_id = "eval-" + question_dict.get("id", str(uuid4()))
        config = {"configurable": {"thread_id": thread_id}}

        # time and generate the response
        response_time_start = perf_counter()
        response = await agent.ainvoke({"messages": question_dict["question"]}, config=config)
        response_time = perf_counter() - response_time_start

        # Create metadata record
        metadata_record = await create_metadata_record(
            question_id=question_dict.get("id"),
            question=question_dict.get("question"),
            expected_answer=question_dict.get("answer"),
            agent_response=response,
            model=model,
            available_tools=tools,
            response_time=response_time,
        )

        # Create agent response record
        agent_response_record = await create_agent_response_record(
            question_id=question_dict.get("id"),
            question=question_dict.get("question"),
            expected_answer=question_dict.get("answer"),
            agent_final_answer=response["messages"][-1].content,
            model=model,
        )

        # Update record lists
        metadata_records.append(metadata_record)
        agent_response_records.append(agent_response_record)

        print(f"Completed evaluation for question: {question_dict.get('question')}")

    except Exception as e:
        print(f"Error: {e}")

        failed_response_record = create_failed_response_record(
            question_id=question_dict.get("id"),
            question=question_dict.get("question"),
            expected_answer=question_dict.get("answer"),
            error=str(e),
        )

        failed_response_records.append(failed_response_record)



async def _evaluate_single_batch(
    agent: CompiledStateGraph,
    batch: list[QuestionRecord],
    prompt: str,
    tools: list[StructuredTool],
    metadata_records: list[MetadataTableRecord],
    agent_response_records: list[AgentResponseTableRecord],
    failed_response_records: list[FailedResponseTableRecord],
    model: str = "openai:gpt-4.1",
) -> None:
    """
    Evaluate a batch of questions asynchronously.

    Parameters
    ----------
    batch : list[QuestionRecord]
        A list of question records containing the question, expected answer and the question id.

    Returns
    -------
    None
        The metadata and agent response records are updated in place.
    """

    tasks = [
        evaluate_single_question(
            agent, question_dict, tools, metadata_records, agent_response_records, failed_response_records, model
        )
        for question_dict in batch
    ]
    return await asyncio.gather(*tasks)


async def _evaluate_batches(
    agent: CompiledStateGraph,
    questions: list[QuestionRecord],
    prompt: str,
    tools: list[StructuredTool],
    metadata_records: list[MetadataTableRecord],
    agent_response_records: list[AgentResponseTableRecord],
    failed_response_records: list[FailedResponseTableRecord],
    model: str = "openai:gpt-4.1",
    batch_size: int = 10,
) -> None:
    """
    Evaluate questions in batches.

    Parameters
    ----------
    questions : list[QuestionRecord]
        A list of question records containing the question, expected answer and the question id.
    prompt : str
        The system prompt to use.
    tools : list[StructuredTool]
        The tools to use.
    metadata_records: list[MetadataTableRecord]
        A list of metadata records to store the metadata for the agent responses.
    agent_response_records: list[AgentResponseTableRecord]
        A list of agent response records to store the agent responses.
    model : str
        The model to use.
    batch_size : int
        The number of questions to process in each batch.

    Returns
    -------
    None
        The metadata and agent response records are updated in place.
    """

    results = list()
    for batch_idx, i in enumerate(range(0, len(questions), batch_size)):
        print(
            f"Processing batch {batch_idx + 1} of {ceil(len(questions) / (batch_size))}  \n",
            end="\r",
        )
        if i + batch_size >= len(questions):
            batch = questions[i:]
        else:
            batch = questions[i : i + batch_size]
        batch_results = await _evaluate_single_batch(
            agent, batch, prompt, tools, metadata_records, agent_response_records, failed_response_records, model
        )

        # Add extracted records to the results list
        results.extend(batch_results)

    return results


async def main():
    """
    Main function to run the agent evaluation.

    Based on the documentation:
    https://github.com/langchain-ai/langchain-mcp-adapters?tab=readme-ov-file#client
    """

    questions = get_questions_from_yaml("questions.yaml")[:3]
    print(f"Retrieved {len(questions)} questions for evaluation.")

    metadata_records = list()
    agent_response_records = list()
    failed_response_records = list()
    async with stdio_client(neo4j_cypher_mcp) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            mcp_tools = await load_mcp_tools(session)

            # We only need to get schema and execute read queries from the Cypher MCP server
            allowed_tools = [
                tool for tool in mcp_tools if tool.name in {"get_neo4j_schema", "read_neo4j_cypher"}
            ]

            # We can also add non-mcp tools for our agent to use
            allowed_tools.append(find_movie_recommendations_tool)

            prompt = get_movies_system_prompt()

            model = "openai:gpt-4.1"
            batch_size = 5

            agent = create_react_agent(
                model=model,
                pre_model_hook=pre_model_hook,
                checkpointer=InMemorySaver(),
                tools=allowed_tools,
                prompt=prompt,
            )

            await _evaluate_batches(
                agent,
                questions,
                prompt,
                allowed_tools,
                metadata_records,
                agent_response_records,
                failed_response_records,
                model,
                batch_size,
            )

            directory = (
                f"{evals_loc}eval_run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )

            os.makedirs(directory, exist_ok=True)

            metadata_df = pd.DataFrame(metadata_records)
            agent_response_df = pd.DataFrame(agent_response_records)

            metadata_df.to_csv(
                f"{directory}/metadata.csv",
                index=False,
            )

            agent_response_df.to_csv(
                f"{directory}/agent_response.csv",
                index=False,
            )

            failed_response_df = pd.DataFrame(failed_response_records, columns=list(FailedResponseTableRecord.__annotations__.keys()))
            failed_response_df.to_csv(
                f"{directory}/failed_response.csv",
                index=False,
            )


if __name__ == "__main__":
    asyncio.run(main())
