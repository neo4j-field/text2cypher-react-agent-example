import yaml
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.prebuilt.chat_agent_executor import AgentState


def get_questions_from_yaml(file_path: str) -> list[dict[str, str]]:
    """
    Get the questions from the yaml file.

    Parameters
    ----------
    file_path: str
        The path to the yaml file.

    Returns
    -------
    list[dict[str, str]]
        A list of dictionaries with the following keys:
        - question: str -> The question to answer.
        - answer: str -> The expected answer to the question.
        - id: str -> The id of the question.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file).get("questions", [])


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
