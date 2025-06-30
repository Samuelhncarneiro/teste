# app/services/job_service.py
import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from app.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

class JobService:
    """Serviço para gerenciamento de jobs de processamento"""
    
    def __init__(self):
        """Inicializa o serviço de jobs"""
        self.jobs = {} 
    
    def create_job(self, file_path: str, filename: str, job_id: Optional[str] = None) -> str:
        """
        Cria um novo job de processamento
        
        Args:
            file_path: Caminho do arquivo
            filename: Nome do arquivo
            job_id: ID opcional do job (se não fornecido, será gerado um UUID)
            
        Returns:
            str: ID do job criado
        """
        if not job_id:
            job_id = str(uuid.uuid4())
        
        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 0.0,
            "file_path": file_path,
            "filename": filename,
            "created_at": datetime.now().isoformat(),
            "model_results": {},
        }
        
        return job_id
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Recupera um job pelo ID
        
        Args:
            job_id: ID do job
            
        Returns:
            Dict: Dados do job ou None se não encontrado
        """
        return self.jobs.get(job_id)
    
    def list_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Lista todos os jobs
        
        Returns:
            Dict: Dicionário com todos os jobs
        """
        return {
            job_id: {
                "status": job["status"],
                "progress": job["progress"],
                "filename": job["filename"],
                "created_at": job["created_at"],
                "models_used": list(job["model_results"].keys()),
            }
            for job_id, job in self.jobs.items()
        }
    
    def update_job_progress(self, job_id: str) -> None:
            """
            Atualiza o progresso geral do job com base nos resultados dos modelos
            
            Args:
                job_id: ID do job a ser atualizado
            """
            if job_id not in self.jobs:
                logger.warning(f"Tentativa de atualizar job inexistente: {job_id}")
                return
                
            job = self.jobs[job_id]
            model_results = job["model_results"]
            
            # Calcular progresso geral - média dos progressos dos modelos
            total_progress = sum(mr.get("progress", 0) for mr in model_results.values())
            if len(model_results) > 0:
                job["progress"] = total_progress / len(model_results)
            
            # Verificar se todos os modelos foram processados
            all_completed = all(mr.get("status") in ["completed", "failed"] for mr in model_results.values())
            
            if all_completed:
                job["status"] = "completed"