# app/services/document_service.py
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import asyncio
from PIL import Image

from app.config import TEMP_DIR, CONVERTED_DIR
from app.utils.file_utils import convert_pdf_to_images, optimize_image
from app.extractors.base import BaseExtractor
from app.services.job_service import JobService

logger = logging.getLogger(__name__)

class DocumentService:
    """Serviço para processamento de documentos"""
    
    def __init__(self, job_service: JobService):
        """
        Inicializa o serviço de documentos
        
        Args:
            job_service: Serviço para gerenciamento de jobs
        """
        self.job_service = job_service
    
    async def process_document(
        self,
        file_path: str,
        filename: str,
        extractor: BaseExtractor,
        job_id: Optional[str] = None
    ) -> str:
        """
        Processa um documento usando o extrator especificado
        
        Args:
            file_path: Caminho para o arquivo
            filename: Nome do arquivo
            extractor: Extrator a ser utilizado
            job_id: ID do job opcional (para permitir IDs personalizados)
            
        Returns:
            str: ID do job criado
        """
        # Criar o job com ID opcional
        if job_id:
            self.job_service.create_job(file_path, filename, job_id)
        else:
            job_id = self.job_service.create_job(file_path, filename)
            
        logger.info(f"Iniciado processamento do documento '{filename}' com ID de job: {job_id}")
        
        # Iniciar processamento em background
        asyncio.create_task(self._process_document_task(file_path, job_id, extractor))
        
        return job_id
        
    async def _process_document_task(
        self,
        file_path: str,
        job_id: str,
        extractor: BaseExtractor
    ) -> None:
        """
        Tarefa de processamento do documento em background
        
        Args:
            file_path: Caminho para o arquivo
            job_id: ID do job
            extractor: Extrator a ser utilizado
        """
        try:
            # Acessar o dicionário de jobs
            jobs_store = self.job_service.jobs
            
            # Executar extração
            await extractor.extract_document(
                file_path, 
                job_id, 
                jobs_store, 
                self.job_service.update_job_progress
            )
            
        except Exception as e:
            logger.exception(f"Erro no processamento do documento: {str(e)}")
            
            # Atualizar job com erro
            if job_id in jobs_store:
                jobs_store[job_id]["status"] = "failed"
                jobs_store[job_id]["error"] = str(e)