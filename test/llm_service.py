# rag_vllm_app/app/services/llm_service.py
import time
from typing import Dict, List, Optional

from app.config import settings
from loguru import logger
from openai import OpenAI


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

                # Если модель не найдена, то используем первую доступную
                if (
                    self.model_name != "local-model"
                    and self.model_name not in model_names
                ):
                    available_model = model_names[0] if model_names else "local-model"
                    logger.warning(
                        f"Модель {self.model_name} не найдена. Используем: {available_model}"
                    )
                    self.model_name = available_model

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
                        "Не удалось подключиться к LMStudio после нескольких попыток"
                    )
                    logger.info(
                        "Убедитесь, что LMStudio запущен и сервер доступен по адресу: "
                        + settings.LMSTUDIO_BASE_URL
                    )
                    break

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

        system_prompt = """Ты - полезный AI ассистент. Отвечай на вопросы пользователя на основе предоставленного контекста.
    Если в контексте нет информации для ответа, просто скажи что не знаешь."""

        if context and len(context) > 0:
            context_text = "\n".join([f"- {doc}" for doc in context[:3]])
            user_prompt = f"""На основе следующей информации ответь на вопрос:

    Информация:
    {context_text}

    Вопрос: {question}

    Ответ:"""
        else:
            # Если контекста нет, отвечаем без него
            user_prompt = f"Вопрос: {question}\n\nОтвет:"

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Добавляем историю диалога если есть
        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_prompt})

        try:
            print(f"DEBUG: Отправляем запрос к LMStudio с {len(messages)} сообщениями")
            print(f"DEBUG: Контекст: {context[:2] if context else 'нет'}")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                stream=False,
            )

            result = response.choices[0].message.content.strip()
            print(f"DEBUG: Получен ответ: {result[:100]}...")
            return result

        except Exception as e:
            print(f"ERROR: Ошибка при запросе к LMStudio: {e}")
            return f"Извините, произошла ошибка при обращении к LMM: {str(e)}"


# Глобальный экземпляр сервиса
llm_service = LMStudioService()
