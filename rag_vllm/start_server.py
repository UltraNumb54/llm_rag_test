import uvicorn
from app.config import settings

if __name__ == "__main__":
    print("Запуск RAG сервера...")
    print(f"Адрес: http://{settings.HOST}:{settings.PORT}")
    print(f"Веб-интерфейс: http://{settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
