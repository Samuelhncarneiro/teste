# app/main.py
import os
import json
import logging
import time
import math
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
from datetime import datetime
from urllib.parse import unquote

# Imports originais mantidos
from app.config import (
    APP_TITLE, APP_DESCRIPTION, APP_VERSION, 
    TEMP_DIR, RESULTS_DIR, CONVERTED_DIR, DATA_DIR,
    CLEANUP_INTERVAL_HOURS, TEMP_RETENTION_HOURS, RESULTS_RETENTION_HOURS,
    LOG_FORMAT, LOG_LEVEL
)

try:
    from app.utils.json_utils import safe_json_dump, fix_nan_in_products, sanitize_for_json
    has_json_utils = True
except ImportError:
    has_json_utils = False
    logger.warning("Módulo json_utils não encontrado, usar serialização padrão")

from app.models.schemas import JobStatus
from app.services.job_service import JobService
from app.services.cleanup_service import init_cleanup_service, get_cleanup_service
from app.services.document_service import DocumentService
from app.extractors.gemini_extractor import GeminiExtractor

# Sistema de métricas simples integrado
class SimpleMetrics:
    """Sistema de métricas simples integrado"""
    
    def __init__(self):
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "total_products_extracted": 0,
            "active_jobs": 0,
            "peak_active_jobs": 0,
            "start_time": datetime.now()
        }
        self.recent_requests = []  # Para rate de sucesso recente
    
    def record_request_start(self, job_id: str):
        """Registra início de processamento"""
        self.stats["total_requests"] += 1
        self.stats["active_jobs"] += 1
        self.stats["peak_active_jobs"] = max(self.stats["peak_active_jobs"], self.stats["active_jobs"])
        
        # Manter apenas últimas 100 requests para rate recente
        if len(self.recent_requests) > 100:
            self.recent_requests.pop(0)
    
    def record_request_success(self, job_id: str, processing_time: float, products_count: int):
        """Registra sucesso no processamento"""
        self.stats["active_jobs"] -= 1
        self.stats["successful_requests"] += 1
        self.stats["total_processing_time"] += processing_time
        self.stats["total_products_extracted"] += products_count
        
        self.recent_requests.append({"success": True, "time": datetime.now()})
    
    def record_request_failure(self, job_id: str, error_type: str = "unknown"):
        """Registra falha no processamento"""
        self.stats["active_jobs"] -= 1
        self.stats["failed_requests"] += 1
        
        self.recent_requests.append({"success": False, "time": datetime.now()})
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas atuais"""
        uptime = datetime.now() - self.stats["start_time"]
        
        # Calcular rate de sucesso recente (últimas 50 requests)
        recent_50 = self.recent_requests[-50:] if len(self.recent_requests) >= 50 else self.recent_requests
        recent_success_rate = sum(1 for r in recent_50 if r["success"]) / len(recent_50) if recent_50 else 1.0
        
        return {
            "processing": {
                "total_requests": self.stats["total_requests"],
                "successful_requests": self.stats["successful_requests"],
                "failed_requests": self.stats["failed_requests"],
                "success_rate": self.stats["successful_requests"] / max(1, self.stats["total_requests"]),
                "recent_success_rate": recent_success_rate,
                "average_processing_time": self.stats["total_processing_time"] / max(1, self.stats["successful_requests"]),
                "total_products_extracted": self.stats["total_products_extracted"]
            },
            "system": {
                "active_jobs": self.stats["active_jobs"],
                "peak_active_jobs": self.stats["peak_active_jobs"],
                "uptime_seconds": int(uptime.total_seconds())
            },
            "timestamp": datetime.now().isoformat()
        }

# Sistema de health check simples
class SimpleHealthChecker:
    """Health checker simples integrado"""
    
    def __init__(self, metrics: SimpleMetrics):
        self.metrics = metrics
    
    def check_health(self) -> Dict[str, Any]:
        """Verifica saúde geral do sistema"""
        stats = self.metrics.get_current_stats()
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Check de sistema
        system_check = self._check_system()
        health_data["checks"]["system"] = system_check
        
        # Check de performance
        performance_check = self._check_performance(stats)
        health_data["checks"]["performance"] = performance_check
        
        # Check de configuração
        config_check = self._check_configuration()
        health_data["checks"]["configuration"] = config_check
        
        # Determinar status geral
        all_checks = [system_check, performance_check, config_check]
        if any(check["status"] == "critical" for check in all_checks):
            health_data["status"] = "critical"
        elif any(check["status"] == "warning" for check in all_checks):
            health_data["status"] = "warning"
        
        return health_data
    
    def _check_system(self) -> Dict[str, Any]:
        """Verifica recursos do sistema"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            issues = []
            
            if memory.percent > 90:
                status = "critical"
                issues.append("Memória crítica")
            elif memory.percent > 80:
                status = "warning"
                issues.append("Memória alta")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 95:
                status = "critical"
                issues.append("Disco crítico")
            elif disk_percent > 90:
                status = "warning"
                issues.append("Disco cheio")
            
            return {
                "status": status,
                "memory_percent": memory.percent,
                "disk_percent": round(disk_percent, 2),
                "issues": issues
            }
        except ImportError:
            return {
                "status": "warning",
                "message": "psutil não disponível para monitoramento de sistema"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _check_performance(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Verifica performance do sistema"""
        processing_stats = stats["processing"]
        
        success_rate = processing_stats["success_rate"]
        recent_success_rate = processing_stats["recent_success_rate"]
        avg_time = processing_stats["average_processing_time"]
        
        status = "healthy"
        issues = []
        
        if recent_success_rate < 0.5:
            status = "critical"
            issues.append("Taxa de sucesso recente muito baixa")
        elif recent_success_rate < 0.8:
            status = "warning"
            issues.append("Taxa de sucesso recente baixa")
        
        if avg_time > 300:  # 5 minutos
            status = "warning"
            issues.append("Tempo de processamento alto")
        
        return {
            "status": status,
            "success_rate": success_rate,
            "recent_success_rate": recent_success_rate,
            "average_processing_time": avg_time,
            "issues": issues
        }
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Verifica configuração do sistema"""
        issues = []
        
        # Verificar API key
        from app.config import GEMINI_API_KEY
        if not GEMINI_API_KEY:
            issues.append("API key do Gemini não configurada")
        
        # Verificar diretórios
        required_dirs = [TEMP_DIR, RESULTS_DIR, CONVERTED_DIR]
        for directory in required_dirs:
            if not os.path.exists(directory):
                issues.append(f"Diretório não existe: {directory}")
        
        status = "critical" if issues else "healthy"
        
        return {
            "status": status,
            "issues": issues
        }

# Inicializar sistemas
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Inicializar métricas e health check
metrics = SimpleMetrics()
health_checker = SimpleHealthChecker(metrics)

# Inicializar serviços originais
job_service = JobService()
document_service = DocumentService(job_service)

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para métricas
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Log simples de métricas
    processing_time = time.time() - start_time
    if processing_time > 1.0:  # Log apenas requests lentos
        logger.info(f"Slow request: {request.url.path} took {processing_time:.2f}s")
    
    return response

def get_gemini_extractor():
    """Dependência para obter o extrator Gemini"""
    return GeminiExtractor()

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Aplicação a iniciar...")
    
    # Verificar saúde inicial
    health = health_checker.check_health()
    if health["status"] == "critical":
        logger.error("❌ Sistema em estado crítico:")
        for check_name, check_data in health["checks"].items():
            if check_data["status"] == "critical":
                logger.error(f"  - {check_name}: {check_data.get('issues', [])}")
        # Não falhar completamente, apenas avisar
        logger.warning("⚠️ Sistema iniciando mesmo com problemas críticos")
    
    # Verificar diretórios necessários
    for dir_path in [TEMP_DIR, RESULTS_DIR, CONVERTED_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"📁 Diretório criado: {dir_path}")
    
    # Inicializar serviço de limpeza
    cleanup_config = {
        "temp_dirs": [
            {"path": TEMP_DIR, "retention_hours": TEMP_RETENTION_HOURS},
            {"path": CONVERTED_DIR, "retention_hours": RESULTS_RETENTION_HOURS},
            {"path": RESULTS_DIR, "retention_hours": RESULTS_RETENTION_HOURS},
        ],
        "cleanup_interval_hours": CLEANUP_INTERVAL_HOURS,
        "retention_hours": RESULTS_RETENTION_HOURS
    }
    
    cleanup_service = init_cleanup_service(cleanup_config)
    if not cleanup_service.running:
        cleanup_service.start()
        logger.info(f"🧹 Serviço de limpeza automática iniciado (intervalo: {CLEANUP_INTERVAL_HOURS}h)")
    
    logger.info("✅ Aplicação iniciada com sucesso!")

@app.get("/health", summary="Verificar saúde do sistema")
async def health_check():
    """
    🔍 Endpoint de verificação de saúde do sistema.
    Retorna informações detalhadas sobre o estado dos componentes.
    """
    health_data = health_checker.check_health()
    
    status_code = 200
    if health_data["status"] == "critical":
        status_code = 503
    elif health_data["status"] == "warning":
        status_code = 200  # Warning não é erro crítico
    
    return JSONResponse(content=health_data, status_code=status_code)

@app.get("/metrics", summary="Obter métricas do sistema")
async def get_metrics():
    """
    📊 Retorna métricas do sistema.
    """
    try:
        current_stats = metrics.get_current_stats()
        return {
            "current": current_stats,
            "description": "Métricas básicas do sistema de extração"
        }
    except Exception as e:
        logger.error(f"Erro ao obter métricas: {str(e)}")
        return JSONResponse(
            content={"error": "Erro ao obter métricas", "detail": str(e)},
            status_code=500
        )

@app.post("/process", response_model=JobStatus, summary="Processar documento")
async def process_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    gemini_extractor: GeminiExtractor = Depends(get_gemini_extractor)
):
    """
    📄 Processa um documento e extrai informações de produtos.
    
    - **file**: Arquivo PDF para processamento
    
    Retorna o status inicial do job.
    """
    
    # Gerar job_id
    job_id = os.path.basename(file.filename).split('.')[0]
    
    # Registrar início do processamento
    metrics.record_request_start(job_id)
    
    try:
        # Validações básicas
        if not file.filename.lower().endswith('.pdf'):
            metrics.record_request_failure(job_id, "invalid_format")
            raise HTTPException(
                status_code=400, 
                detail="Apenas arquivos PDF são suportados"
            )
        
        # Verificar tamanho do arquivo
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        max_size_mb = 50  # Limite configurável
        if file_size_mb > max_size_mb:
            metrics.record_request_failure(job_id, "file_too_large")
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo muito grande. Máximo: {max_size_mb}MB"
            )
        
        # Salvar arquivo
        file_location = os.path.join(TEMP_DIR, f"{job_id}_{file.filename}")
        
        with open(file_location, "wb") as file_object:
            file_object.write(file_content)
        
        logger.info(f"📄 Arquivo salvo: {file_location} ({file_size_mb:.2f}MB)")
        
        # Processar documento
        job_id = await document_service.process_document(
            file_location, file.filename, gemini_extractor, job_id
        )
        
        # Retornar status inicial
        job = job_service.get_job(job_id)
        return JobStatus(**job)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Tratar outros erros
        logger.exception(f"Erro ao processar documento: {str(e)}")
        metrics.record_request_failure(job_id, "processing_error")
        
        raise HTTPException(
            status_code=500, 
            detail=f"Erro no processamento: {str(e)}"
        )

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    🔍 Retorna o status de um job de processamento.
    
    - **job_id**: ID do job
    """
    logger.info(f"🔍 Consultando status do job: {job_id}")

    job_id = unquote(job_id)
    job = job_service.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404, 
            detail=f"Job não encontrado: {job_id}"
        )
    
    # Se job foi concluído com sucesso, registrar métricas
    if job["status"] == "completed":
        # Extrair informações do resultado
        if "model_results" in job and "gemini" in job["model_results"]:
            result = job["model_results"]["gemini"].get("result", {})
            products_count = len(result.get("products", []))
            processing_time = job["model_results"]["gemini"].get("processing_time", 0)
            
            metrics.record_request_success(job_id, processing_time, products_count)
    elif job["status"] == "failed":
        metrics.record_request_failure(job_id, "job_failed")
    
    return JobStatus(**job)

@app.get("/job/{job_id}/excel", summary="Obter resultado em Excel")
async def get_job_excel(job_id: str, season: str = None):

    job_id = unquote(job_id)
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job ainda em processamento")
    
    # Verificar se temos resultados
    if "gemini" not in job.get("model_results", {}) or "result" not in job.get("model_results", {}).get("gemini", {}):
        logger.warning(f"Job {job_id} não tem resultados gemini disponíveis. Status: {job.get('status')}")
        logger.warning(f"Model results keys: {list(job.get('model_results', {}).keys())}")
        raise HTTPException(status_code=404, detail="Resultados não disponíveis")
    
    try:
        # Extrair dados do resultado
        extraction_result = job["model_results"]["gemini"]["result"]
        
        # Criar DataFrame
        df = create_dataframe_from_extraction(extraction_result, season)
        
        # Gerar arquivo Excel
        excel_path = os.path.join(RESULTS_DIR, f"{job_id}_result.xlsx")
        
        if not os.path.exists(excel_path):
            df.to_excel(excel_path, index=False)
            logger.info(f"📊 Arquivo Excel gerado: {excel_path}")
        
        return FileResponse(
            path=excel_path,
            filename=f"{job_id}_result.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        logger.exception(f"Erro ao gerar Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar Excel: {str(e)}")

@app.get("/job/{job_id}/json", summary="Obter resultado em JSON")
async def get_job_json(job_id: str):
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job ainda em processamento")
    
    if "gemini" not in job["model_results"] or "result" not in job["model_results"]["gemini"]:
        raise HTTPException(status_code=404, detail="Resultados não disponíveis")
    
    try:
        extraction_result = job["model_results"]["gemini"]["result"]
        
        if has_json_utils:
            try:
                from app.utils.recovery_system import ProcessingRecovery
                extraction_result = ProcessingRecovery.fix_extraction_result(extraction_result)
            except ImportError:
                pass
        
        def sanitize_datetime(obj):
            import datetime
            if isinstance(obj, dict):
                return {k: sanitize_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_datetime(item) for item in obj]
            elif isinstance(obj, datetime.datetime):
                return obj.isoformat()
            elif isinstance(obj, datetime.date):
                return obj.isoformat()
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            else:
                return obj
        
        extraction_result = sanitize_datetime(extraction_result)
        
        return JSONResponse(content=extraction_result, status_code=200)
        
    except Exception as e:
        logger.exception(f"Erro ao gerar JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar JSON: {str(e)}")


@app.get("/jobs", summary="Listar todos os jobs")
async def list_jobs():
    """
    📋 Lista todos os jobs ativos no sistema.
    """
    return job_service.list_jobs()

@app.get("/", summary="Status da API")
async def root():
    """
    🏠 Verifica se a API está funcionando e retorna informações básicas.
    """
    try:
        stats = metrics.get_current_stats()
        
        return {
            "message": "API de Extração de Documentos com Melhorias",
            "status": "online",
            "version": APP_VERSION,
            "swagger_ui": "/docs",
            "health_check": "/health",
            "metrics": "/metrics",
            "stats": stats["processing"],
            "uptime_seconds": stats["system"]["uptime_seconds"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erro no endpoint root: {str(e)}")
        return {
            "message": "API de Extração de Documentos",
            "status": "online",
            "version": APP_VERSION,
            "error": "Erro ao obter estatísticas"
        }

def create_dataframe_from_extraction(
    extraction_result: Dict[str, Any], 
    season: Optional[str] = None
) -> pd.DataFrame:
    """
    Cria um DataFrame pandas a partir dos resultados da extração.
    Versão melhorada com tratamento de erros robusto.
    """
    try:
        data = []
        material_code_counts = {}
        
        # Obter informações do pedido
        order_info = extraction_result.get("order_info", {})
        current_season = season or order_info.get("season", "")
        
        # Processar cada produto
        for product in extraction_result.get("products", []):
            product_name = product.get("name", "")
            material_code_base = product.get("material_code", "")
            
            if not material_code_base:
                continue  # Pular produtos sem código
            
            # Gerar código único
            if material_code_base in material_code_counts:
                material_code_counts[material_code_base] += 1
            else:
                material_code_counts[material_code_base] = 1
            
            material_code = f"{material_code_base}.{material_code_counts[material_code_base]}"
            
            category = product.get("category", "")
            model = product.get("model", "")
            brand = product.get("brand", order_info.get("brand", ""))
            supplier = order_info.get("supplier", "")
            
            # Processar cada cor do produto
            for color in product.get("colors", []):
                color_code = color.get("color_code", "")
                color_name = color.get("color_name", "")
                unit_price = color.get("unit_price", 0)
                sales_price = color.get("sales_price", 0)
                
                # Processar cada tamanho da cor
                for size_info in color.get("sizes", []):
                    size = size_info.get("size", "")
                    quantity = size_info.get("quantity", 0)
                    
                    if quantity <= 0:
                        continue  # Pular tamanhos sem quantidade
                    
                    # Adicionar linha ao DataFrame
                    data.append({
                        "Material Code": material_code,
                        "Base Code": material_code_base,
                        "Product Name": product_name,
                        "Category": category,
                        "Model": model,
                        "Color Code": color_code,
                        "Color Name": color_name,
                        "Size": size,
                        "Quantity": quantity,
                        "Unit Price": unit_price,
                        "Sales Price": sales_price,
                        "Brand": brand,
                        "Supplier": supplier,
                        "Season": current_season,
                        "Order Number": order_info.get("order_number", ""),
                        "Date": order_info.get("date", ""),
                        "Document Type": order_info.get("document_type", "")
                    })
        
        # Criar DataFrame
        if data:
            df = pd.DataFrame(data)
            
            # Substituir valores NaN por valores padrão
            numeric_columns = ["Unit Price", "Sales Price", "Quantity"]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Substituir strings vazias
            string_columns = ["Color Code", "Color Name", "Size", "Category"]
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].fillna("")
            
            return df
        else:
            # DataFrame vazio com colunas definidas
            return pd.DataFrame(columns=[
                "Material Code", "Base Code", "Product Name", "Category", "Model",
                "Color Code", "Color Name", "Size", "Quantity",
                "Unit Price", "Sales Price", "Brand", "Supplier",
                "Season", "Order Number", "Date", "Document Type"
            ])
    
    except Exception as e:
        logger.error(f"Erro ao criar DataFrame: {str(e)}")
        # Retornar DataFrame vazio em caso de erro
        return pd.DataFrame()

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 Iniciando Extrator de Documentos (Versão Melhorada)")
    print("="*60)
    print(f"📋 Swagger UI: http://localhost:8000/docs")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print(f"📊 Métricas: http://localhost:8000/metrics")
    print(f"📁 Uploads: {TEMP_DIR}")
    print(f"📁 Resultados: {RESULTS_DIR}")
    print("\n💡 Pressione Ctrl+C para encerrar")
    print("="*60 + "\n")
    
    # Iniciar servidor
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level=LOG_LEVEL.lower()
    )