from openai import OpenAI
from app.config import settings
import torch

class LLMService:
    def __init__(self):
        self.client = OpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY
        )
        self.model_name = settings.LLM_MODEL_NAME
        
    def generate_response(self, prompt, temperature=0.2, max_tokens=1000):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Ошибка генерации ответа: {str(e)}")
    
    def check_connection(self):
        """Проверка подключения к LLM"""
        try:
            self.client.models.list()
            return True
        except:
            return False
