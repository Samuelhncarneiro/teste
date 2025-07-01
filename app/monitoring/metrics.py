# app/monitoring/metrics.py
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Ponto de métrica"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class ProcessingStats:
    """Estatísticas de processamento"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    total_products_extracted: int = 0
    total_pages_processed: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_processing_time(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_processing_time / self.successful_requests
    
    @property
    def products_per_page(self) -> float:
        if self.total_pages_processed == 0:
            return 0.0
        return self.total_products_extracted / self.total_pages_processed

class MetricsCollector:
    """Coletor de métricas da aplicação"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque())
        self.stats = ProcessingStats()
        self.lock = threading.Lock()
        
        # Métricas de performance
        self.active_jobs = 0
        self.peak_active_jobs = 0
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Iniciar limpeza automática
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self._cleanup_thread.start()
    
    def record_request_start(self, job_id: str):
        """Registra início de processamento"""
        with self.lock:
            self.active_jobs += 1
            self.peak_active_jobs = max(self.peak_active_jobs, self.active_jobs)
            self.stats.total_requests += 1
            
            self._add_metric("active_jobs", self.active_jobs)
            self._add_metric("total_requests", self.stats.total_requests)
    
    def record_request_success(
        self, 
        job_id: str, 
        processing_time: float, 
        products_count: int,
        pages_count: int
    ):
        """Registra sucesso no processamento"""
        with self.lock:
            self.active_jobs -= 1
            self.stats.successful_requests += 1
            self.stats.total_processing_time += processing_time
            self.stats.total_products_extracted += products_count
            self.stats.total_pages_processed += pages_count
            
            self._add_metric("active_jobs", self.active_jobs)
            self._add_metric("processing_time", processing_time)
            self._add_metric("products_extracted", products_count)
            self._add_metric("pages_processed", pages_count)
            self._add_metric("success_rate", self.stats.success_rate)
    
    def record_request_failure(self, job_id: str, error_type: str):
        """Registra falha no processamento"""
        with self.lock:
            self.active_jobs -= 1
            self.stats.failed_requests += 1
            self.error_counts[error_type] += 1
            
            self._add_metric("active_jobs", self.active_jobs)
            self._add_metric("failed_requests", self.stats.failed_requests)
            self._add_metric("success_rate", self.stats.success_rate)
    
    def record_api_call(self, provider: str, duration: float, success: bool):
        """Registra chamada de API externa"""
        with self.lock:
            labels = {"provider": provider, "success": str(success)}
            self._add_metric("api_call_duration", duration, labels)
    
    def record_memory_usage(self, usage_mb: float):
        """Registra uso de memória"""
        self._add_metric("memory_usage_mb", usage_mb)
    
    def record_file_processing(self, file_size_mb: float, file_type: str):
        """Registra processamento de arquivo"""
        labels = {"file_type": file_type}
        self._add_metric("file_size_mb", file_size_mb, labels)
    
    def _add_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Adiciona métrica à coleção"""
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        
        self.metrics[name].append(metric_point)
        
        # Limitar tamanho da queue
        max_points = self.retention_hours * 60  # um ponto por minuto
        if len(self.metrics[name]) > max_points:
            self.metrics[name].popleft()
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas atuais"""
        with self.lock:
            return {
                "processing": {
                    "total_requests": self.stats.total_requests,
                    "successful_requests": self.stats.successful_requests,
                    "failed_requests": self.stats.failed_requests,
                    "success_rate": round(self.stats.success_rate, 3),
                    "average_processing_time": round(self.stats.average_processing_time, 2),
                    "products_per_page": round(self.stats.products_per_page, 2)
                },
                "system": {
                    "active_jobs": self.active_jobs,
                    "peak_active_jobs": self.peak_active_jobs,
                    "error_counts": dict(self.error_counts)
                },
                "timestamp": datetime.now().isoformat()
            }
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Retorna resumo de métricas das últimas N horas"""
        cutoff = datetime.now() - timedelta(hours=hours)
        summary = {}
        
        with self.lock:
            for metric_name, points in self.metrics.items():
                recent_points = [p for p in points if p.timestamp > cutoff]
                
                if recent_points:
                    values = [p.value for p in recent_points]
                    summary[metric_name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "latest": values[-1]
                    }
        
        return summary
    
    def export_metrics(self, file_path: str):
        """Exporta métricas para arquivo"""
        export_data = {
            "stats": self.get_current_stats(),
            "metrics_summary": self.get_metrics_summary(24),  # 24 horas
            "export_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Métricas exportadas para: {file_path}")
        except Exception as e:
            logger.error(f"Erro ao exportar métricas: {e}")
    
    def _cleanup_old_metrics(self):
        """Thread para limpeza automática de métricas antigas"""
        while True:
            try:
                time.sleep(3600)  # Executar a cada hora
                
                cutoff = datetime.now() - timedelta(hours=self.retention_hours)
                
                with self.lock:
                    for metric_name, points in self.metrics.items():
                        # Remover pontos antigos
                        while points and points[0].timestamp < cutoff:
                            points.popleft()
                
                logger.debug("Limpeza de métricas antigas concluída")
                
            except Exception as e:
                logger.error(f"Erro na limpeza de métricas: {e}")

class PerformanceMonitor:
    """Monitor de performance para métodos"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_operations: Dict[str, float] = {}
    
    def start_operation(self, operation_name: str) -> str:
        """Inicia monitoramento de operação"""
        operation_id = f"{operation_name}_{int(time.time())}"
        self.active_operations[operation_id] = time.time()
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True):
        """Finaliza monitoramento de operação"""
        if operation_id in self.active_operations:
            duration = time.time() - self.active_operations[operation_id]
            operation_name = operation_id.rsplit('_', 1)[0]
            
            # Registrar métrica
            labels = {"operation": operation_name, "success": str(success)}
            self.metrics._add_metric("operation_duration", duration, labels)
            
            del self.active_operations[operation_id]
            return duration
        return 0

def performance_monitor(operation_name: str):
    """Decorator para monitoramento automático de performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = PerformanceMonitor(metrics_collector)
            operation_id = monitor.start_operation(operation_name)
            
            try:
                result = await func(*args, **kwargs)
                monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                monitor.end_operation(operation_id, success=False)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            monitor = PerformanceMonitor(metrics_collector)
            operation_id = monitor.start_operation(operation_name)
            
            try:
                result = func(*args, **kwargs)
                monitor.end_operation(operation_id, success=True)
                return result
            except Exception as e:
                monitor.end_operation(operation_id, success=False)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Instância global
metrics_collector = MetricsCollector(retention_hours=24)