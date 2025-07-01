# app/health/health_check.py
import psutil
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthChecker:
    """Sistema de verificação de saúde da aplicação"""
    
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
    
    def check_health(self) -> Dict[str, Any]:
        """Verifica saúde geral do sistema"""
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Verificar memória
        memory_check = self._check_memory()
        health_data["checks"]["memory"] = memory_check
        
        # Verificar disco
        disk_check = self._check_disk_space()
        health_data["checks"]["disk"] = disk_check
        
        # Verificar performance
        performance_check = self._check_performance()
        health_data["checks"]["performance"] = performance_check
        
        # Verificar configuração
        config_check = self._check_configuration()
        health_data["checks"]["configuration"] = config_check
        
        # Determinar status geral
        all_checks = [memory_check, disk_check, performance_check, config_check]
        if any(check["status"] == "critical" for check in all_checks):
            health_data["status"] = "critical"
        elif any(check["status"] == "warning" for check in all_checks):
            health_data["status"] = "warning"
        
        return health_data
    
    def _check_memory(self) -> Dict[str, Any]:
        """Verifica uso de memória"""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            status = "healthy"
            if usage_percent > 90:
                status = "critical"
            elif usage_percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "usage_percent": usage_percent,
                "available_gb": round(memory.available / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Verifica espaço em disco"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            status = "healthy"
            if usage_percent > 95:
                status = "critical"
            elif usage_percent > 90:
                status = "warning"
            
            return {
                "status": status,
                "usage_percent": round(usage_percent, 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "total_gb": round(disk.total / (1024**3), 2)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _check_performance(self) -> Dict[str, Any]:
        """Verifica performance do sistema"""
        try:
            stats = self.metrics.get_current_stats()
            processing_stats = stats["processing"]
            
            success_rate = processing_stats["success_rate"]
            avg_time = processing_stats["average_processing_time"]
            
            status = "healthy"
            issues = []
            
            if success_rate < 0.8:
                status = "warning"
                issues.append("Taxa de sucesso baixa")
            
            if success_rate < 0.5:
                status = "critical"
            
            if avg_time > 300:  # 5 minutos
                status = "warning"
                issues.append("Tempo de processamento alto")
            
            return {
                "status": status,
                "success_rate": success_rate,
                "average_processing_time": avg_time,
                "issues": issues
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Verifica configuração do sistema"""
        try:
            from app.config.settings import config
            
            issues = []
            
            # Verificar API key
            if not config.api.gemini_api_key:
                issues.append("API key do Gemini não configurada")
            
            # Verificar diretórios
            import os
            required_dirs = [
                config.get_full_path(config.storage.temp_dir),
                config.get_full_path(config.storage.results_dir),
                config.get_full_path(config.storage.converted_dir)
            ]
            
            for directory in required_dirs:
                if not os.path.exists(directory):
                    issues.append(f"Diretório não existe: {directory}")
            
            status = "critical" if issues else "healthy"
            
            return {
                "status": status,
                "issues": issues
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Instância global
health_checker = HealthChecker(metrics_collector)