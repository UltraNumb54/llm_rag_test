# rag_vllm_app/app/routers/files.py:
import os

import aiofiles
from app.config import settings
from app.services.file_processor import FileProcessor
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from loguru import logger

router = APIRouter()
file_processor = FileProcessor()


@router.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Проверяем размер файла
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком большой. Максимальный размер: {settings.MAX_FILE_SIZE // (1024 * 1024)}MB",
            )

        # Сохраняем файл
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        background_tasks.add_task(file_processor.process_file, file_path, file.filename)

        return {
            "status": "success",
            "filename": file.filename,
            "message": "Файл загружен и отправлен на обработку",
        }

    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def list_files():
    """Список загруженных файлов"""
    try:
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            return {"files": []}

        files = []
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            stats = os.stat(file_path)
            files.append(
                {
                    "filename": filename,
                    "size": stats.st_size,
                    "uploaded_at": stats.st_ctime,
                }
            )

        return {"files": files}
    except Exception as e:
        logger.error(f"Ошибка получения списка файлов: {e}")
        raise HTTPException(status_code=500, detail=str(e))
