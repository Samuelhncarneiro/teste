# app/config/settings.py
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import yaml
import json

# Carregar variáveis de ambiente
load_dotenv()

@dataclass
class ExtractorSettings:
    """Configurações do extrator"""
    max_retries: int = 3
    timeout_seconds: int = 30
    confidence_threshold: float = 0.7
    enable_color_mapping: bool = True
    enable_barcode_generation: bool = True
    enable_supplier_detection: bool = True
    default_markup: float = 2.73
    
@dataclass
class ApiSettings:
    """Configurações de APIs"""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # segundos
    
@dataclass
class StorageSettings:
    """Configurações de armazenamento"""
    temp_dir: str = "temp_uploads"
    results_dir: str = "results"
    converted_dir: str = "converted_images"
    cleanup_interval_hours: int = 6
    temp_retention_hours: int = 24
    results_retention_hours: int = 72
    max_file_size_mb: int = 50
    
@dataclass
class ProcessingSettings:
    """Configurações de processamento"""
    max_concurrent_jobs: int = 5
    max_pages_per_document: int = 50
    image_quality: int = 85
    image_max_dimension: int = 1200
    pdf_dpi: int = 150
    
@dataclass
class LoggingSettings:
    """Configurações de logging"""
    level: str = "INFO"
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path: Optional[str] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    enable_console: bool = True
    
@dataclass
class MonitoringSettings:
    """Configurações de monitoramento"""
    enable_metrics: bool = True
    metrics_retention_days: int = 30
    alert_on_error_rate: float = 0.1  # 10%
    alert_on_processing_time: int = 300  # 5 minutos
    health_check_interval: int = 60  # segundos

class AppConfig:
    """Configuração centralizada da aplicação"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.config_file = config_file or os.path.join(self.base_dir, "config.yaml")
        
        # Inicializar com defaults
        self.extractor = ExtractorSettings()
        self.api = ApiSettings()
        self.storage = StorageSettings()
        self.processing = ProcessingSettings()
        self.logging = LoggingSettings()
        self.monitoring = MonitoringSettings()
        
        # Carregar configurações
        self._load_from_env()
        self._load_from_file()
        self._validate_and_setup_directories()
    
    def _load_from_env(self):
        """Carrega configurações das variáveis de ambiente"""
        
        # API Settings
        self.api.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.api.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        
        # Storage Settings
        self.storage.cleanup_interval_hours = int(os.getenv("CLEANUP_INTERVAL_HOURS", "6"))
        self.storage.temp_retention_hours = int(os.getenv("TEMP_RETENTION_HOURS", "24"))
        self.storage.results_retention_hours = int(os.getenv("RESULTS_RETENTION_HOURS", "72"))
        
        # Processing Settings
        self.processing.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "5"))
        
        # Logging Settings
        self.logging.level = os.getenv("LOG_LEVEL", "INFO")
        self.logging.file_path = os.getenv("LOG_FILE_PATH")
    
    def _load_from_file(self):
        """Carrega configurações do arquivo YAML"""
        if not os.path.exists(self.config_file):
            self._create_default_config_file()
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                self._update_from_dict(config_data)
                
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
    
    def _create_default_config_file(self):
        """Cria arquivo de configuração padrão"""
        default_config = {
            'extractor': {
                'max_retries': self.extractor.max_retries,
                'timeout_seconds': self.extractor.timeout_seconds,
                'confidence_threshold': self.extractor.confidence_threshold,
                'default_markup': self.extractor.default_markup
            },
            'storage': {
                'cleanup_interval_hours': self.storage.cleanup_interval_hours,
                'temp_retention_hours': self.storage.temp_retention_hours,
                'results_retention_hours': self.storage.results_retention_hours,
                'max_file_size_mb': self.storage.max_file_size_mb
            },
            'processing': {
                'max_concurrent_jobs': self.processing.max_concurrent_jobs,
                'max_pages_per_document': self.processing.max_pages_per_document,
                'image_quality': self.processing.image_quality
            },
            'monitoring': {
                'enable_metrics': self.monitoring.enable_metrics,
                'metrics_retention_days': self.monitoring.metrics_retention_days
            }
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Erro ao criar arquivo de configuração: {e}")
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Atualiza configurações a partir de dicionário"""
        
        if 'extractor' in config_data:
            ext_config = config_data['extractor']
            for key, value in ext_config.items():
                if hasattr(self.extractor, key):
                    setattr(self.extractor, key, value)
        
        if 'storage' in config_data:
            storage_config = config_data['storage']
            for key, value in storage_config.items():
                if hasattr(self.storage, key):
                    setattr(self.storage, key, value)
        
        if 'processing' in config_data:
            proc_config = config_data['processing']
            for key, value in proc_config.items():
                if hasattr(self.processing, key):
                    setattr(self.processing, key, value)
        
        if 'monitoring' in config_data:
            mon_config = config_data['monitoring']
            for key, value in mon_config.items():
                if hasattr(self.monitoring, key):
                    setattr(self.monitoring, key, value)
    
    def _validate_and_setup_directories(self):
        """Valida configurações e cria diretórios necessários"""
        
        # Validar API key
        if not self.api.gemini_api_key:
            raise ValueError("GEMINI_API_KEY é obrigatória")
        
        # Criar diretórios
        directories = [
            os.path.join(self.base_dir, self.storage.temp_dir),
            os.path.join(self.base_dir, self.storage.results_dir),
            os.path.join(self.base_dir, self.storage.converted_dir)
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_full_path(self, relative_path: str) -> str:
        """Retorna caminho completo baseado no diretório base"""
        return os.path.join(self.base_dir, relative_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário"""
        return {
            'extractor': self.extractor.__dict__,
            'api': {k: v for k, v in self.api.__dict__.items() if k != 'gemini_api_key'},
            'storage': self.storage.__dict__,
            'processing': self.processing.__dict__,
            'logging': self.logging.__dict__,
            'monitoring': self.monitoring.__dict__
        }
    
    def reload(self):
        """Recarrega configurações"""
        self._load_from_env()
        self._load_from_file()

# Instância global
config = AppConfig()

# Exports para compatibilidade com código existente
GEMINI_API_KEY = config.api.gemini_api_key
GEMINI_MODEL = config.api.gemini_model
TEMP_DIR = config.get_full_path(config.storage.temp_dir)
RESULTS_DIR = config.get_full_path(config.storage.results_dir)
CONVERTED_DIR = config.get_full_path(config.storage.converted_dir)