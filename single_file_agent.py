"""This is a single file containing all the code to create a simple agent that can answer questions about the movie database."""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from neo4j import GraphDatabase, RoutingControl
from pydantic import BaseModel, Field

if load_dotenv():
    print("Loaded .env file")
else:
    print("No .env file found")


def find_movie_recommendations(
    movie_title: str, min_user_rating: float = 4.0, limit: int = 10
) -> list[dict[str, Any]]:
    """
    Search the database movie recommendations based on movie title and rating criteria.
    """

    query = """
MATCH (target:Movie)
WHERE target.title = $movieTitle
MATCH (target)<-[r1:RATED]-(u:User)
WHERE r1.rating >= $minRating
MATCH (u)-[r2:RATED]->(similar:Movie)
WHERE similar <> target 
  AND r2.rating >= $minRating 
  AND similar.imdbRating IS NOT NULL
WITH similar, count(*) as supporters, avg(r2.rating) as avgRating
WHERE supporters >= 10
RETURN similar.title, similar.year, similar.imdbRating, 
       supporters as people_who_loved_both, 
       round(avgRating, 2) as avg_rating_by_target_lovers
ORDER BY supporters DESC, avgRating DESC
LIMIT $limit
    """

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    results = driver.execute_query(
        query,
        parameters_={"movieTitle": movie_title, "minRating": min_user_rating, "limit": limit},
        database_=os.getenv("NEO4J_DATABASE"),
        routing_=RoutingControl.READ,
        result_transformer_=lambda r: r.data(),
    )
    return results


class FindMovieRecommendationsInput(BaseModel):
    movie_title: str = Field(
        ...,
        description="The title of the movie to find recommendations for. If beginning with 'The', then will follow format of 'Title, The'.",
    )
    min_user_rating: float = Field(
        default=4.0,
        description="The minimum rating of the movie to find recommendations for. ",
        ge=0.5,
        le=5.0,
    )
    limit: int = Field(
        default=10,
        description="The maximum number of recommendations to return. ",
        ge=1,
    )


find_movie_recommendations_tool = StructuredTool.from_function(
    func=find_movie_recommendations,  #           -> The function that the tool calls when executed
    # name=...,                                   -> this is populated by the function name
    # description=...,                            -> this is populated by the function docstring
    args_schema=FindMovieRecommendationsInput,  # -> The input schema for the tool
    return_direct=False,  #                       -> Whether to return the raw result to the user
    # coroutine=...,                              -> An async version of the function
)

# The Neo4j Cypher MCP server will be used to get the database schema and execute Cypher queries
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

CONFIG = {"configurable": {"thread_id": "1"}}

SYSTEM_PROMPT = """You are a Neo4j expert that knows how to write Cypher queries to address movie questions.
As a Cypher expert, when writing queries:
* You must always ensure you have the data model schema to inform your queries
* If an error is returned from the database, you may refactor your query or ask the user to provide additional information
* If an empty result is returned, use your best judgement to determine if the query is correct.

If using a tool that does NOT require writing a Cypher query, you do not need the database schema.

As a well respected movie expert:
* Ensure that you provide detailed responses with citations to the underlying data"""


def pre_model_hook(state: AgentState) -> dict[str, list[AnyMessage]]:
    """
    This function will be called every time before the node that calls LLM.

    Documentation:
    https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/?h=create_react_agent

    Parameters
    ----------
    state : AgentState
        The state of the agent.

    Returns
    -------
    dict[str, list[AnyMessage]]
        The updated messages to pass to the LLM as context.
    """

    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=30_000,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,  # -> We always want to include the system prompt in the context
    )
    # You can return updated messages either under:
    # `llm_input_messages` -> To keep the original message history unmodified in the graph state and pass the updated history only as the input to the LLM
    # `messages`           -> To overwrite the original message history in the graph state with the updated history
    return {"llm_input_messages": trimmed_messages}


async def print_astream(async_stream, output_messages_key: str = "llm_input_messages") -> None:
    """
    Print the stream of messages from the agent.

    Based on the documentation:
    https://langchain-ai.github.io/langgraph/how-tos/create-react-agent-manage-message-history/?h=create_react_agent#keep-the-original-message-history-unmodified

    Parameters
    ----------
    async_stream : AsyncGenerator[dict[str, dict[str, list[AnyMessage]]], None]
        The stream of messages from the agent.
    output_messages_key : str, optional
        The key to use for the output messages, by default "llm_input_messages".
    """

    async for chunk in async_stream:
        for node, update in chunk.items():
            print(f"Update from node: {node}")
            messages_key = output_messages_key if node == "pre_model_hook" else "messages"
            for message in update[messages_key]:
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()

        print("\n\n")


async def main():
    """
    Main function to run the agent.

    Based on the documentation:
    https://github.com/langchain-ai/langchain-mcp-adapters?tab=readme-ov-file#client
    """

    # start up the MCP server locally and run our agent
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

            # Create and run the agent
            agent = create_react_agent(
                "openai:gpt-4.1",  #              -> The model to use
                allowed_tools,  #                 -> The tools to use
                pre_model_hook=pre_model_hook,  # -> The function to call before the model is called
                checkpointer=InMemorySaver(),  #  -> The checkpoint to use
                prompt=SYSTEM_PROMPT,  #          -> The system prompt to use
            )

            # conversation loop
            print(
                "\n===================================== Chat =====================================\n"
            )

            while True:
                user_input = input("> ")
                if user_input.lower() in {"exit", "quit", "q"}:
                    break

                await print_astream(
                    agent.astream({"messages": user_input}, config=CONFIG, stream_mode="updates")
                )


if __name__ == "__main__":
    asyncio.run(main())
