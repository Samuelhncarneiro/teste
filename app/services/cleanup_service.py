# app/services/cleanup_service.py
import os
import logging
import time
import shutil
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class CleanupService:
    """
    Serviço para limpeza automática de diretórios temporários após um período definido
    """
    
    def __init__(
        self,
        temp_dirs=None,
        cleanup_interval_hours=24,  
        retention_hours=72,         
    ):
        """
        Inicializa o serviço de limpeza
        
        Args:
            temp_dirs: Lista de diretórios para limpar (dict com 'path' e 'retention_hours')
            cleanup_interval_hours: Intervalo entre verificações em horas
            retention_hours: Tempo padrão de retenção em horas
        """
        self.default_retention_hours = retention_hours
        self.cleanup_interval_hours = cleanup_interval_hours
        
        # Definir diretórios padrão se não forem fornecidos
        if temp_dirs is None:
            self.temp_dirs = [
                {"path": "temp_uploads", "retention_hours": 24},  # arquivos de upload temporários: 1 dia
                {"path": "converted_images", "retention_hours": 48},  # imagens convertidas: 2 dias
                {"path": "results", "retention_hours": 72},  # resultados: 3 dias
            ]
        else:
            self.temp_dirs = temp_dirs
            
        # Flag para controlar o loop de limpeza
        self.running = False
        # Thread para executar limpeza em segundo plano
        self.cleanup_thread = None
        
        logger.info(f"Serviço de limpeza iniciado. Verificações a cada {cleanup_interval_hours} horas.")
    
    def start(self):
        """Inicia o serviço de limpeza em segundo plano"""
        if self.running:
            logger.warning("Serviço de limpeza já está em execução")
            return
            
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Thread de limpeza iniciada")
    
    def stop(self):
        """Para o serviço de limpeza"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
            logger.info("Serviço de limpeza parado")
    
    def _cleanup_loop(self):
        """Loop principal de limpeza que executa periodicamente"""
        while self.running:
            try:
                self.run_cleanup()
            except Exception as e:
                logger.error(f"Erro durante execução da limpeza: {str(e)}")
                
            # Esperar pelo próximo intervalo de limpeza
            interval_seconds = self.cleanup_interval_hours * 3600
            # Dividir o tempo em pequenos intervalos para permitir interrupção mais rápida
            for _ in range(int(interval_seconds / 10)):
                if not self.running:
                    break
                time.sleep(10)
    
    def run_cleanup(self):
        """Executa a rotina de limpeza para todos os diretórios configurados"""
        start_time = time.time()
        total_removed = 0
        
        logger.info("Iniciando rotina de limpeza de arquivos temporários")
        
        for dir_config in self.temp_dirs:
            dir_path = dir_config["path"]
            retention_hours = dir_config.get("retention_hours", self.default_retention_hours)
            
            # Verificar se o diretório existe
            if not os.path.exists(dir_path):
                logger.info(f"Diretório {dir_path} não existe, pulando")
                continue
                
            # Limpar arquivos antigos neste diretório
            try:
                removed_count = self._cleanup_directory(dir_path, retention_hours)
                total_removed += removed_count
                logger.info(f"Limpeza de {dir_path}: {removed_count} arquivos removidos (retenção: {retention_hours}h)")
            except Exception as e:
                logger.error(f"Erro ao limpar diretório {dir_path}: {str(e)}")
        
        duration = time.time() - start_time
        logger.info(f"Limpeza concluída em {duration:.1f}s. Total de {total_removed} arquivos removidos.")
    
    def _cleanup_directory(self, dir_path, retention_hours):
        """
        Limpa arquivos mais antigos que o período de retenção em um diretório
        
        Args:
            dir_path: Caminho do diretório
            retention_hours: Tempo de retenção em horas
            
        Returns:
            int: Número de arquivos removidos
        """
        # Calcular timestamp limite (agora - período de retenção)
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        cutoff_timestamp = cutoff_time.timestamp()
        
        removed_count = 0
        
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            
            # Obter hora da última modificação
            try:
                mtime = os.path.getmtime(item_path)
            except (FileNotFoundError, PermissionError):
                continue
                
            # Verificar se é mais antigo que o período de retenção
            if mtime < cutoff_timestamp:
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    removed_count += 1
                except (PermissionError, OSError) as e:
                    logger.warning(f"Não foi possível remover {item_path}: {str(e)}")
        
        return removed_count
    
    def clean_specific_job(self, job_id):
        """
        Limpa todos os arquivos associados a um job específico
        
        Args:
            job_id: ID do job para limpar
            
        Returns:
            int: Número de arquivos removidos
        """
        total_removed = 0
        logger.info(f"Limpando arquivos do job {job_id}")
        
        for dir_config in self.temp_dirs:
            dir_path = dir_config["path"]
            
            # Verificar se o diretório existe
            if not os.path.exists(dir_path):
                continue
                
            # Procurar por arquivos com este job_id
            for item in os.listdir(dir_path):
                if job_id in item:
                    item_path = os.path.join(dir_path, item)
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        total_removed += 1
                        logger.info(f"Removido: {item_path}")
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Não foi possível remover {item_path}: {str(e)}")
        
        return total_removed

# Instância global do serviço de limpeza
cleanup_service = None

def init_cleanup_service(config=None):
    """
    Inicializa o serviço de limpeza global
    
    Args:
        config: Configuração para o serviço de limpeza
    """
    global cleanup_service
    
    if cleanup_service is None:
        cleanup_service = CleanupService(**(config or {}))
        cleanup_service.start()
        
    return cleanup_service

def get_cleanup_service():
    """
    Obtém a instância global do serviço de limpeza
    
    Returns:
        CleanupService: Instância do serviço
    """
    global cleanup_service
    
    if cleanup_service is None:
        cleanup_service = init_cleanup_service()
        
    return cleanup_service