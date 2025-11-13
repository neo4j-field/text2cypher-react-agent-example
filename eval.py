import asyncio
import os
from datetime import datetime
from math import ceil
from time import perf_counter
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics.collections import AnswerRelevancy, RougeScore

from evals.models import QuestionRecord, ResponseTableRecord
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

# we will use this LLM as a judge in our evaluations that don't require structured output
evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Setup LLM and embeddings for answer relevancy - this eval requires structured output from the evaluator LLM
client = AsyncOpenAI()
evaluator_llm_structured_output = llm_factory("gpt-4o-mini", client=client)
embeddings = embedding_factory(
    "openai", model="text-embedding-3-small", client=client, interface="modern"
)

# https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/traditional/#rouge-score
# look at the longest common subsequence between the reference and the response
# use F1 score to measure the quality of the match
rouge_scorer = RougeScore(rouge_type="rougeL", mode="fmeasure")

# https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/#factual-correctness
factual_correctness_scorer = FactualCorrectness(
    llm=evaluator_llm, mode="F1", atomicity="high", coverage="high"
)

# https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/#answer-relevancy
answer_relevancy_scorer = AnswerRelevancy(
    llm=evaluator_llm_structured_output, embeddings=embeddings
)


async def evaluate_single_question(
    agent: CompiledStateGraph,
    question_dict: dict[str, str],
    tools: list[StructuredTool],
    model: str = "openai:gpt-4.1",
) -> ResponseTableRecord:
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

        response_time_start = perf_counter()
        response = await agent.ainvoke({"messages": question_dict["question"]}, config=config)
        response_time = perf_counter() - response_time_start

        tool_calls = [
            tool_call
            for message in response["messages"]
            if isinstance(message, AIMessage)
            and hasattr(message, "tool_calls")
            and message.tool_calls
            for tool_call in message.tool_calls
        ]

        # capture all text2cypher queries
        cyphers = [c.get("args") for c in tool_calls if c.get("name") == "read_neo4j_cypher"]

        rouge_score = await rouge_scorer.ascore(
            reference=question_dict.get("answer"), response=response["messages"][-1].content
        )

        sample = SingleTurnSample(
            response=response["messages"][-1].content,
            reference=question_dict.get("answer"),
        )
        # we have to use the SingleTurnSample for factual correctness
        factual_correctness_score = await factual_correctness_scorer.single_turn_ascore(sample)

        answer_relevancy_score = await answer_relevancy_scorer.ascore(
            user_input=question_dict.get("question"), response=response["messages"][-1].content
        )

        print(f"Completed evaluation for question: {question_dict.get('question')}")
        return ResponseTableRecord(
            question_id=question_dict.get("id"),
            question=question_dict.get("question"),
            expected_answer=question_dict.get("answer"),
            agent_final_answer=response["messages"][-1].content,
            generated_cypher=cyphers,
            model=model,
            available_tools=[t.name for t in tools],
            called_tools=tool_calls,
            num_messages=len(response["messages"]),
            num_llm_calls=len([m for m in response["messages"] if isinstance(m, AIMessage)]),
            num_tool_calls=len(tool_calls),
            response_time=response_time,
            error=None,
            rouge_f1_score=rouge_score.value,
            factual_correctness_f1_score=factual_correctness_score,
            answer_relevancy_score=answer_relevancy_score.value,
        )

    except Exception as e:
        print(f"Error: {e}")
        return ResponseTableRecord(
            question_id=question_dict.get("id"),
            question=question_dict.get("question"),
            expected_answer=question_dict.get("answer"),
            agent_final_answer=None,
            generated_cypher=list(),
            model=model,
            available_tools=[t.name for t in tools],
            called_tools=list(),
            num_messages=None,
            num_llm_calls=None,
            num_tool_calls=None,
            response_time=None,
            error=str(e),
            rouge_f1_score=None,
            factual_correctness_f1_score=None,
            answer_relevancy_score=None,
        )


async def _evaluate_single_batch(
    agent: CompiledStateGraph,
    batch: list[QuestionRecord],
    prompt: str,
    tools: list[StructuredTool],
    model: str = "openai:gpt-4.1",
) -> list[ResponseTableRecord]:
    """
    Evaluate a batch of questions asynchronously.

    Parameters
    ----------
    batch : list[QuestionRecord]
        A list of question records containing the question, expected answer and the question id.

    Returns
    -------
    list[ResponseTableRecord]
        A list of response table records containing the agent response and associated metadata.
    """

    tasks = [
        evaluate_single_question(agent, question_dict, tools, model) for question_dict in batch
    ]
    return await asyncio.gather(*tasks)


async def _evaluate_batches(
    agent: CompiledStateGraph,
    questions: list[QuestionRecord],
    prompt: str,
    tools: list[StructuredTool],
    model: str = "openai:gpt-4.1",
    batch_size: int = 10,
) -> list[ResponseTableRecord]:
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
    model : str
        The model to use.
    batch_size : int
        The number of questions to process in each batch.

    Returns
    -------
    list[ResponseTableRecord]
        A list of response table records containing the agent response and associated metadata.
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
        batch_results = await _evaluate_single_batch(agent, batch, prompt, tools, model)

        # Add extracted records to the results list
        results.extend(batch_results)

    return results


async def main():
    """
    Main function to run the agent evaluation.

    Based on the documentation:
    https://github.com/langchain-ai/langchain-mcp-adapters?tab=readme-ov-file#client
    """

    questions = get_questions_from_yaml("questions.yaml")
    print(f"Retrieved {len(questions)} questions for evaluation.")

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

            eval_results = await _evaluate_batches(
                agent, questions, prompt, allowed_tools, model, batch_size
            )

            df = pd.DataFrame(eval_results)

            df.to_csv(
                f"{evals_loc}eval_benchmark_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
                index=False,
            )


if __name__ == "__main__":
    asyncio.run(main())
