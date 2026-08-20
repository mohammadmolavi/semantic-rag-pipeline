from .models import Question


def generate_answer(question: Question, *, top_k: int = 4) -> Question:
    from rag.services import ask_question, serialize_sources

    source = question.document.vector_source if question.document_id else None
    result = ask_question(question.question, source=source, top_k=top_k)
    question.answer = result.answer
    question.sources = serialize_sources(result.sources)
    question.save(update_fields=["answer", "sources"])
    return question
