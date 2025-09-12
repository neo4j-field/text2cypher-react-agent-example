from typing import TypedDict, Optional

class ResponseTableRecord(TypedDict):
    question_id: str
    question: str
    expected_answer: str
    agent_final_answer: Optional[str]
    generated_cypher: list[str]
    model: str
    available_tools: list[str]
    called_tools: list[str]
    num_messages: Optional[int]
    num_llm_calls: Optional[int]
    num_tool_calls: Optional[int]
    response_time: Optional[float]
    error: Optional[str]

class QuestionRecord(TypedDict):
    id: Optional[str]
    question: str
    answer: Optional[str]


               