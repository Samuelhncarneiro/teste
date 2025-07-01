# app/utils/error_handler.py
import logging
import traceback
from typing import Dict, Any, Optional, Callable
from functools import wraps
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCode(Enum):
    # Erros de entrada
    INVALID_FILE_FORMAT = "E001"
    FILE_TOO_LARGE = "E002"
    FILE_CORRUPTED = "E003"
    
    # Erros de processamento
    EXTRACTION_FAILED = "E101"
    PARSING_ERROR = "E102"
    VALIDATION_ERROR = "E103"
    
    # Erros de API
    API_RATE_LIMIT = "E201"
    API_AUTHENTICATION = "E202"
    API_TIMEOUT = "E203"
    
    # Erros de sistema
    STORAGE_ERROR = "E301"
    MEMORY_ERROR = "E302"
    CONFIGURATION_ERROR = "E303"

@dataclass
class ErrorInfo:
    code: ErrorCode
    message: str
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ErrorHandler:
    """Sistema de tratamento de erros centralizado"""
    
    def __init__(self):
        self.error_callbacks: List[Callable] = []
        self.error_history: List[ErrorInfo] = []
        self.max_history = 1000
    
    def add_callback(self, callback: Callable[[ErrorInfo], None]):
        """Adiciona callback para tratamento de erros"""
        self.error_callbacks.append(callback)
    
    def handle_error(
        self, 
        error: Exception, 
        code: ErrorCode,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Dict[str, Any] = None,
        suggestions: List[str] = None
    ) -> ErrorInfo:
        """Trata erro de forma padronizada"""
        
        error_info = ErrorInfo(
            code=code,
            message=str(error),
            severity=severity,
            context=context or {},
            suggestions=suggestions or []
        )
        
        # Log apropriado baseado na severidade
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"[{code.value}] {error_info.message}", exc_info=True)
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"[{code.value}] {error_info.message}", exc_info=True)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"[{code.value}] {error_info.message}")
        else:
            logger.info(f"[{code.value}] {error_info.message}")
        
        # Executar callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.error(f"Erro em callback de erro: {e}")
        
        # Armazenar no histórico
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        return error_info
    
    def create_response(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Cria resposta padronizada de erro"""
        return {
            "success": False,
            "error": {
                "code": error_info.code.value,
                "message": error_info.message,
                "severity": error_info.severity.value,
                "suggestions": error_info.suggestions,
                "timestamp": error_info.timestamp
            }
        }

def handle_exceptions(
    code: ErrorCode, 
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    default_return=None
):
    """Decorator para tratamento automático de exceções"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, code, severity)
                return default_return
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, code, severity)
                return default_return
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Instância global
error_handler = ErrorHandler()