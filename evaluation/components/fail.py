from evaluation.models import FailedResponseTableRecord

def create_failed_response_record(
    question_id: str,
    question: str,
    expected_answer: str,
    error: str,
) -> FailedResponseTableRecord:
    return FailedResponseTableRecord(
        question_id=question_id,
        question=question,
        expected_answer=expected_answer,
        error=error,
    )