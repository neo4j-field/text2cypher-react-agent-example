import asyncio
from datetime import datetime
from math import ceil
from time import perf_counter

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import StructuredTool
from tools.find_movie_recommendations import find_movie_recommendations_tool
from langgraph.prebuilt import create_react_agent
from evals.models import ResponseTableRecord, QuestionRecord
from prompt import get_movies_system_prompt
from utils import get_questions_from_yaml, pre_model_hook


if load_dotenv():
    print("Loaded .env file")
else:
    print("No .env file found")

evals_loc = "evals/output/"
eval_results = list()

async def evaluate_single_question(question_dict: dict[str, str], prompt: str, tools: list[StructuredTool], model: str = "openai:gpt-4.1") -> ResponseTableRecord:
    """
    Initialize a fresh agent and evaluate a single question.
    """
    try:
        assert question_dict.get("question") is not None, "Question not found"
        # Create and run the agent
        # use the default tools and prompt for our custom agent
        agent_config = create_react_agent(
            model=model,
            pre_model_hook=pre_model_hook,
            checkpointer=InMemorySaver(),
            tools=tools,
            prompt=prompt,
        )

        response_time_start = perf_counter()
        response = await agent_config.ainvoke({"messages": question_dict["question"]})
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
        )


async def _evaluate_single_batch(batch: list[QuestionRecord], prompt: str, tools: list[StructuredTool], model: str = "openai:gpt-4.1") -> list[ResponseTableRecord]:
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
    
    tasks = [evaluate_single_question(question_dict, prompt, tools, model) for question_dict in batch]
    return await asyncio.gather(*tasks)


async def _evaluate_batches(questions: list[QuestionRecord], prompt: str, tools: list[StructuredTool], model: str = "openai:gpt-4.1", batch_size: int = 10) -> list[ResponseTableRecord]:
    """
    Create embeddings for a Pandas DataFrame of text chunks in batches.

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
        print(f"Processing batch {batch_idx+1} of {ceil(len(questions)/(batch_size))}  \n", end="\r") 
        if i + batch_size >= len(questions):
            batch = questions[i:]
        else:
            batch = questions[i:i+batch_size]
        batch_results = await _evaluate_single_batch(batch, prompt, tools, model)

        # Add extracted records to the results list
        results.extend(batch_results)

    return results


async def main():
    """
    Main function to run the agent.

    Based on the documentation:
    https://github.com/langchain-ai/langchain-mcp-adapters?tab=readme-ov-file#client
    """

  
    questions = get_questions_from_yaml(f"questions.yaml")
    print(f"Retrieved {len(questions)} questions for evaluation.")

    prompt = get_movies_system_prompt()
    tools = [find_movie_recommendations_tool]
    model = "openai:gpt-4.1"
    batch_size = 10

    eval_results = await _evaluate_batches(questions, prompt, tools, model, batch_size)

    df = pd.DataFrame(eval_results)

    df.to_csv(
        f"{evals_loc}eval_benchmark_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        index=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
