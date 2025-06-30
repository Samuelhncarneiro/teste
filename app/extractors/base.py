# app/extractors/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseExtractor(ABC):
    """Classe base abstrata para todos os extratores"""
    
    @abstractmethod
    async def analyze_context(self, document_path: str) -> str:
        """
        Analisa o contexto geral do documento
        
        Args:
            document_path: Caminho para o documento
            
        Returns:
            str: Descrição contextual do documento
        """
        pass
    
    @abstractmethod
    async def process_page(
        self, 
        image_path: str, 
        context: str,
        page_number: int,
        total_pages: int,
        previous_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processa uma página do documento
        
        Args:
            image_path: Caminho para a imagem da página
            context: Contexto do documento
            page_number: Número da página atual
            total_pages: Total de páginas no documento
            previous_result: Resultados de páginas anteriores
            
        Returns:
            Dict: Resultado da extração para a página
        """
        pass
    
    @abstractmethod
    async def extract_document(
        self, 
        document_path: str,
        job_id: str,
        jobs_store: Dict[str, Any],
        update_progress_callback
    ) -> Dict[str, Any]:
        """
        Extrai informações de um documento completo
        
        Args:
            document_path: Caminho para o documento
            job_id: ID do job em processamento
            jobs_store: Armazenamento de jobs
            update_progress_callback: Callback para atualizar progresso
            
        Returns:
            Dict: Resultado completo da extração
        """
        pass