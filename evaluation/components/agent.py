from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics._factual_correctness import FactualCorrectness
from ragas.metrics.collections import AnswerRelevancy, RougeScore
from dotenv import load_dotenv
from evaluation.models import AgentResponseTableRecord

if load_dotenv():
    print("Loaded .env file")
else:
    print("No .env file found")

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


async def create_agent_response_record(
    question_id: str,
    question: str,
    expected_answer: str,
    agent_final_answer: str,
    model: str,
) -> AgentResponseTableRecord:
    """
    Create an agent response record for the agent response.
    """

    try:
        rouge_score = await rouge_scorer.ascore(reference=expected_answer, response=agent_final_answer)

        sample = SingleTurnSample(
            response=agent_final_answer,
            reference=expected_answer,
        )
        # we have to use the SingleTurnSample for factual correctness
        factual_correctness_score = await factual_correctness_scorer.single_turn_ascore(sample)

        answer_relevancy_score = await answer_relevancy_scorer.ascore(
            user_input=question, response=agent_final_answer
        )

        return AgentResponseTableRecord(
            question_id=question_id,
            question=question,
            expected_answer=expected_answer,
            agent_final_answer=agent_final_answer,
            model=model,
            rouge_f1_score=rouge_score.value,
            factual_correctness_f1_score=factual_correctness_score,
            answer_relevancy_score=answer_relevancy_score.value,
            error=None,
        )
    except Exception as e:
        return AgentResponseTableRecord(
            question_id=question_id,
            question=question,
            expected_answer=expected_answer,
            agent_final_answer=None,
            model=model,
            rouge_f1_score=None,
            factual_correctness_f1_score=None,
            answer_relevancy_score=None,
            error=str(e),
        )
