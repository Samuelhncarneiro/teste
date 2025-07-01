# app/main.py
import os
import json
import logging
import asyncio
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
from datetime import datetime

# Imports das melhorias
from app.config.settings import config
from app.utils.error_handler import error_handler, handle_exceptions, ErrorCode, ErrorSeverity
from app.monitoring.metrics import metrics_collector, performance_monitor, health_checker
from app.monitoring.health_check import health_checker
from app.extractors.unified_extractor import UnifiedExtractor, ExtractionConfig, ExtractionStrategy

# Imports originais mantidos
from app.models.schemas import JobStatus
from app.services.job_service import JobService
from app.services.cleanup_service import init_cleanup_service, get_cleanup_service
from app.services.document_service import DocumentService

# Configurar logging baseado na configuração
logging.basicConfig(
    level=getattr(logging, config.logging.level), 
    format=config.logging.format
)
logger = logging.getLogger(__name__)

# Callback para monitoramento de erros
def error_callback(error_info):
    """Callback para tratamento de erros"""
    metrics_collector.record_request_failure("unknown", error_info.code.value)
    
    # Aqui você pode adicionar integração com sistemas de alertas
    if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
        logger.critical(f"Erro crítico detectado: {error_info.code.value} - {error_info.message}")

error_handler.add_callback(error_callback)

# Inicializar serviços
job_service = JobService()

# Configurar extrator unificado
extraction_config = ExtractionConfig(
    strategy=ExtractionStrategy.AUTO,
    enable_color_mapping=config.extractor.enable_color_mapping,
    enable_barcode_generation=config.extractor.enable_barcode_generation,
    enable_supplier_detection=config.extractor.enable_supplier_detection,
    max_retries=config.extractor.max_retries,
    timeout_seconds=config.extractor.timeout_seconds,
    confidence_threshold=config.extractor.confidence_threshold
)

unified_extractor = UnifiedExtractor(extraction_config)
document_service = DocumentService(job_service)

# Inicializar FastAPI
app = FastAPI(
    title="Extrator de Documentos Avançado",
    description="API otimizada para extrair informações de documentos usando IA",
    version="2.0.0",
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
    
    # Registrar métricas
    processing_time = time.time() - start_time
    metrics_collector.record_api_call(
        provider="internal",
        duration=processing_time,
        success=response.status_code < 400
    )
    
    return response

def get_unified_extractor():
    """Dependência para obter o extrator unificado"""
    return unified_extractor

@app.on_event("startup")
async def startup_event():
    """Evento executado na inicialização do aplicativo"""
    logger.info("Aplicação a iniciar...")
    
    # Verificar saúde inicial
    health = health_checker.check_health()
    if health["status"] == "critical":
        logger.error("Sistema em estado crítico:")
        for check_name, check_data in health["checks"].items():
            if check_data["status"] == "critical":
                logger.error(f"  - {check_name}: {check_data}")
        raise RuntimeError("Sistema não pode iniciar em estado crítico")
    
    # Inicializar serviço de limpeza
    cleanup_config = {
        "temp_dirs": [
            {"path": config.storage.temp_dir, "retention_hours": config.storage.temp_retention_hours},
            {"path": config.storage.converted_dir, "retention_hours": config.storage.results_retention_hours},
            {"path": config.storage.results_dir, "retention_hours": config.storage.results_retention_hours},
        ],
        "cleanup_interval_hours": config.storage.cleanup_interval_hours,
        "retention_hours": config.storage.results_retention_hours
    }
    
    cleanup_service = init_cleanup_service(cleanup_config)
    if not cleanup_service.running:
        cleanup_service.start()
        logger.info(f"🧹 Serviço de limpeza automática iniciado (intervalo: {config.storage.cleanup_interval_hours}h)")
    
    logger.info("Aplicação iniciada com sucesso!")

@app.get("/health", summary="Verificar saúde do sistema")
async def health_check():
    """
    Endpoint de verificação de saúde do sistema.
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
async def get_metrics(hours: int = 1):
    """
    Retorna métricas do sistema das últimas N horas.
    
    - **hours**: Número de horas para análise (padrão: 1)
    """
    try:
        current_stats = metrics_collector.get_current_stats()
        metrics_summary = metrics_collector.get_metrics_summary(hours)
        
        return {
            "current": current_stats,
            "summary": metrics_summary,
            "period_hours": hours
        }
    except Exception as e:
        error_info = error_handler.handle_error(
            e, ErrorCode.CONFIGURATION_ERROR, ErrorSeverity.MEDIUM
        )
        return error_handler.create_response(error_info)

@app.post("/process", response_model=JobStatus, summary="Processar documento")
@performance_monitor("document_processing")
async def process_document(
    file: UploadFile = File(...),
    strategy: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    extractor: UnifiedExtractor = Depends(get_unified_extractor)
):
    """
    Processa um documento e extrai informações de produtos.
    
    - **file**: Arquivo PDF para processamento
    - **strategy**: Estratégia de extração (auto, tabular, sequential, hybrid)
    
    Retorna o status inicial do job.
    """
    
    # Registrar início do processamento
    job_id = os.path.basename(file.filename).split('.')[0]
    metrics_collector.record_request_start(job_id)
    
    try:
        # Validações de entrada
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Apenas arquivos PDF são suportados"
            )
        
        # Verificar tamanho do arquivo
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > config.storage.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo muito grande. Máximo: {config.storage.max_file_size_mb}MB"
            )
        
        # Registrar métricas do arquivo
        metrics_collector.record_file_processing(file_size_mb, "pdf")
        
        # Salvar arquivo
        file_location = os.path.join(
            config.get_full_path(config.storage.temp_dir), 
            f"{job_id}_{file.filename}"
        )
        
        with open(file_location, "wb") as file_object:
            file_object.write(file_content)
        
        logger.info(f"Ficheiro salvo: {file_location} ({file_size_mb:.2f}MB)")
        
        # Configurar estratégia se especificada
        if strategy:
            try:
                extraction_strategy = ExtractionStrategy(strategy.lower())
                extractor.config.strategy = extraction_strategy
                logger.info(f"Estratégia definida: {strategy}")
            except ValueError:
                logger.warning(f"Estratégia inválida '{strategy}', usando AUTO")
        
        # Processar documento
        job_id = await document_service.process_document(
            file_location, file.filename, extractor, job_id
        )
        
        # Retornar status inicial
        job = job_service.get_job(job_id)
        return JobStatus(**job)
    
    except HTTPException:
        # Re-raise HTTP exceptions
        metrics_collector.record_request_failure(job_id, "validation_error")
        raise
    except Exception as e:
        # Tratar outros erros
        error_info = error_handler.handle_error(
            e, ErrorCode.EXTRACTION_FAILED, ErrorSeverity.HIGH,
            context={"job_id": job_id, "filename": file.filename}
        )
        metrics_collector.record_request_failure(job_id, error_info.code.value)
        
        raise HTTPException(
            status_code=500, 
            detail=error_info.message
        )

@app.get("/job/{job_id}", response_model=JobStatus)
@handle_exceptions(ErrorCode.EXTRACTION_FAILED, ErrorSeverity.LOW)
async def get_job_status(job_id: str):
    """
    Retorna o status de um job de processamento.
    
    - **job_id**: ID do job
    """
    logger.info(f"Status do job: {job_id}")
    
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
            
            # Estimar páginas processadas (pode ser melhorado)
            pages_count = result.get("_metadata", {}).get("pages_processed", 1)
            
            metrics_collector.record_request_success(
                job_id, processing_time, products_count, pages_count
            )
    
    return JobStatus(**job)

@app.get("/job/{job_id}/excel", summary="Obter resultado em Excel")
@handle_exceptions(ErrorCode.EXTRACTION_FAILED, ErrorSeverity.MEDIUM)
async def get_job_excel(job_id: str, season: str = None):
    """
    Retorna os resultados do job em formato Excel.
    
    - **job_id**: ID do job
    - **season**: Temporada (opcional, ex: "FW23")
    """
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job ainda em processamento")
    
    # Verificar se temos resultados
    if "gemini" not in job["model_results"] or "result" not in job["model_results"]["gemini"]:
        raise HTTPException(status_code=404, detail="Resultados não disponíveis")
    
    try:
        # Extrair dados do resultado
        extraction_result = job["model_results"]["gemini"]["result"]
        
        # Criar DataFrame
        df = create_dataframe_from_extraction(extraction_result, season)
        
        # Gerar arquivo Excel
        excel_path = os.path.join(
            config.get_full_path(config.storage.results_dir), 
            f"{job_id}_result.xlsx"
        )
        
        if not os.path.exists(excel_path):
            df.to_excel(excel_path, index=False)
            logger.info(f"Arquivo Excel gerado: {excel_path}")
        
        return FileResponse(
            path=excel_path,
            filename=f"{job_id}_result.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        error_info = error_handler.handle_error(
            e, ErrorCode.STORAGE_ERROR, ErrorSeverity.MEDIUM
        )
        raise HTTPException(status_code=500, detail=error_info.message)

@app.get("/job/{job_id}/json", summary="Obter resultado em JSON")
@handle_exceptions(ErrorCode.EXTRACTION_FAILED, ErrorSeverity.MEDIUM)
async def get_job_json(job_id: str):
    """
    Retorna os resultados do job em formato JSON.
    
    - **job_id**: ID do job
    """
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job ainda em processamento")
    
    if "gemini" not in job["model_results"] or "result" not in job["model_results"]["gemini"]:
        raise HTTPException(status_code=404, detail="Resultados não disponíveis")
    
    extraction_result = job["model_results"]["gemini"]["result"]
    return JSONResponse(content=extraction_result, status_code=200)

@app.get("/jobs", summary="Listar todos os jobs")
async def list_jobs():
    """
    Lista todos os jobs ativos no sistema.
    """
    return job_service.list_jobs()

@app.get("/config", summary="Obter configurações")
async def get_config():
    """
    Retorna as configurações atuais do sistema (sem dados sensíveis).
    """
    return config.to_dict()

@app.post("/config/reload", summary="Recarregar configurações")
async def reload_config():
    """
    Recarrega as configurações do sistema.
    """
    try:
        config.reload()
        logger.info("🔄 Configurações recarregadas")
        return {"message": "Configurações recarregadas com sucesso"}
    except Exception as e:
        error_info = error_handler.handle_error(
            e, ErrorCode.CONFIGURATION_ERROR, ErrorSeverity.MEDIUM
        )
        raise HTTPException(status_code=500, detail=error_info.message)

@app.get("/", summary="Status da API")
async def root():
    """
    Verifica se a API está funcionando e retorna informações básicas.
    """
    stats = metrics_collector.get_current_stats()
    
    return {
        "message": "API de Extração de Documentos v2.0",
        "status": "online",
        "version": "2.0.0",
        "swagger_ui": "/docs",
        "health_check": "/health",
        "metrics": "/metrics",
        "stats": stats["processing"],
        "timestamp": datetime.now().isoformat()
    }

def create_dataframe_from_extraction(
    extraction_result: Dict[str, Any], 
    season: Optional[str] = None
) -> pd.DataFrame:
    """
    Cria um DataFrame pandas a partir dos resultados da extração.
    Versão otimizada com tratamento de erros melhorado.
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
            df = df.fillna({
                "Unit Price": 0.0,
                "Sales Price": 0.0,
                "Quantity": 0,
                "Color Code": "",
                "Color Name": "",
                "Size": "",
                "Category": "ACESSÓRIOS"
            })
            
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
    print("🚀 Iniciando Extrator de Documentos v2.0")
    print("="*60)
    print(f"📋 Swagger UI: http://localhost:8000/docs")
    print(f"❤️  Health Check: http://localhost:8000/health")
    print(f"📊 Métricas: http://localhost:8000/metrics")
    print(f"📁 Uploads: {config.get_full_path(config.storage.temp_dir)}")
    print(f"📁 Resultados: {config.get_full_path(config.storage.results_dir)}")
    print("="*60 + "\n")
    
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level=config.logging.level.lower()
    )