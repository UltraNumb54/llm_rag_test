# rag_vllm_app/app/routers/chat.py:
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.llm_service import llm_service
from app.services.vector_store import vector_store
from app.services.reranker import reranker_service
from rag_vllm_app.app.config import settings
from loguru import logger

router = APIRouter()

# Временное хранилище для истории диалогов (в продакшене заменить на Redis/БД)
conversation_store = {}


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    use_reranking: bool = True


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[str]
    suggested_questions: List[str]


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Создаем или получаем ID диалога
        conversation_id = (
            request.conversation_id or f"conv_{len(conversation_store) + 1}"
        )

        # Получаем историю диалога
        history = conversation_store.get(conversation_id, [])

        # Ищем релевантные документы
        search_results = vector_store.search(request.message, top_k=settings.TOP_K)

        if not search_results:
            response = (
                "Извините, в моей базе знаний нет информации для ответа на этот вопрос."
            )
            sources = []
        else:
            # Извлекаем тексты документов
            documents = [result["document"] for result in search_results]

            # Реранкинг если включен
            if request.use_reranking and len(documents) > 1:
                reranked_docs = reranker_service.rerank(
                    request.message, documents, top_k=settings.RERANK_TOP_K
                )
                context_docs = [doc for doc, score in reranked_docs]
            else:
                context_docs = documents[: settings.RERANK_TOP_K]

            # Генерируем ответ с контекстом
            response = llm_service.generate_with_context(
                question=request.message,
                context=context_docs,
                conversation_history=history,
            )

            sources = [
                result["document"][:100] + "..." for result in search_results[:3]
            ]

        # Обновляем историю диалога
        history.extend(
            [
                {"role": "user", "content": request.message},
                {"role": "assistant", "content": response},
            ]
        )

        # Ограничиваем историю последними 10 сообщениями
        conversation_store[conversation_id] = history[-10:]

        # Предлагаемые вопросы
        suggested_questions = [
            "Какие документы можно загружать?",
            "Как работает поиск?",
            "Какие форматы файлов поддерживаются?",
        ]

        return ChatResponse(
            response=response,
            conversation_id=conversation_id,
            sources=sources,
            suggested_questions=suggested_questions,
        )

    except Exception as e:
        logger.error(f"Ошибка в чате: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Получить историю диалога"""
    history = conversation_store.get(conversation_id, [])
    return {"conversation_id": conversation_id, "history": history}
