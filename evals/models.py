from typing import Any, Optional, TypedDict


class ReadNeo4jCypherToolInput(TypedDict):
    query: str
    params: Optional[dict[str, Any]]


class ResponseTableRecord(TypedDict):
    "A record created for the response table. The contents of this record may be used for further evaluation."

    question_id: str
    question: str
    expected_answer: str
    agent_final_answer: Optional[str]
    generated_cypher: list[ReadNeo4jCypherToolInput]
    model: str
    available_tools: list[str]
    called_tools: list[str]
    num_messages: Optional[int]
    num_llm_calls: Optional[int]
    num_tool_calls: Optional[int]
    response_time: Optional[float]
    error: Optional[str]


class QuestionRecord(TypedDict):
    "A record read from the questions yaml file."

    id: Optional[str]
    question: str
    answer: Optional[str]
