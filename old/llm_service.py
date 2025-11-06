# rag_vllm_app/app/services/llm_service.py:

from openai import OpenAI
from rag_vllm_app.app.config import settings
from loguru import logger
import time
from typing import List, Dict, Optional


class LMStudioService:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.LMSTUDIO_BASE_URL, api_key=settings.LMSTUDIO_API_KEY
        )
        self.model_name = settings.LMSTUDIO_MODEL_NAME
        self.test_connection()

    def test_connection(self):
        """Тестирование подключения к LMStudio"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Проверяем доступность моделей
                models = self.client.models.list()
                model_names = [model.id for model in models.data]
                logger.info(f"Доступные модели в LMStudio: {model_names}")

                # Если модель не найдена, используем первую доступную
                if (
                    self.model_name != "local-model"
                    and self.model_name not in model_names
                ):
                    available_model = model_names[0] if model_names else "local-model"
                    logger.warning(
                        f"Модель {self.model_name} не найдена. Используем: {available_model}"
                    )
                    self.model_name = available_model

                # Тестовый запрос
                test_response = self.generate("Ответь 'OK'", max_tokens=5)
                logger.success(
                    f"LMStudio подключен успешно. Тестовый ответ: {test_response}"
                )
                return

            except Exception as e:
                logger.warning(
                    f"Попытка {attempt + 1}/{max_retries}: Ошибка подключения к LMStudio: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    logger.error(
                        "❌ Не удалось подключиться к LMStudio после нескольких попыток"
                    )
                    logger.info(
                        "Убедитесь, что LMStudio запущен и сервер доступен по адресу: "
                        + settings.LMSTUDIO_BASE_URL
                    )
                    raise

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system_message: Optional[str] = None,
    ) -> str:
        """Генерация ответа через LMStudio"""
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ошибка генерации через LMStudio: {e}")
            return f"Ошибка при обращении к модели: {str(e)}"

    def generate_with_context(
        self,
        question: str,
        context: List[str],
        conversation_history: Optional[List[Dict]] = None,
    ) -> str:
        """Генерация ответа с контекстом RAG"""
        system_prompt = """Ты - AI ассистент технической поддержки. Отвечай на вопросы пользователя ТОЛЬКО на основе предоставленного контекста.
Если в контексте нет информации для ответа, вежливо сообщи об этом и предложи обратиться к специалисту.

Важные правила:
1. Отвечай только на основе предоставленного контекста
2. Будь точным и полезным
3. Если информации недостаточно, так и скажи
4. Не придумывай информацию"""

        context_text = "\n\n".join(
            [f"[Документ {i + 1}]: {doc}" for i, doc in enumerate(context)]
        )

        user_prompt = f"""Контекст для ответа:
{context_text}

Вопрос пользователя: {question}

Ответь максимально полезно и точно на основе контекста выше:"""

        messages = [{"role": "system", "content": system_prompt}]

        # Добавляем историю диалога если есть
        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ошибка генерации с контекстом: {e}")
            return "Извините, произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже."


# Глобальный экземпляр сервиса
llm_service = LMStudioService()
