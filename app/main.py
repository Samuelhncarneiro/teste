# app/main.py
import os
import json
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
from datetime import datetime

from app.config import (
    APP_TITLE, APP_DESCRIPTION, APP_VERSION, 
    TEMP_DIR, RESULTS_DIR, CONVERTED_DIR, DATA_DIR,
    CLEANUP_INTERVAL_HOURS, TEMP_RETENTION_HOURS, RESULTS_RETENTION_HOURS,
    LOG_FORMAT, LOG_LEVEL
)

from app.models.schemas import JobStatus
from app.services.job_service import JobService
from app.services.cleanup_service import init_cleanup_service, get_cleanup_service
from app.services.document_service import DocumentService
from app.extractors.gemini_extractor import GeminiExtractor
from app.utils.integration import initialize_recovery_features
from app.utils.recovery_system import ProcessingRecovery

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

try:
    from app.utils.integration import initialize_recovery_features
    has_recovery_system = True
except ImportError:
    has_recovery_system = False
    logger.warning("Sistema de recuperação não encontrado, operando sem proteção contra valores NaN")

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

def get_gemini_extractor():
    """
    Dependência para obter o extrator Gemini avançado
    """
    return GeminiExtractor()

@app.on_event("startup")
async def startup_event():
    """Evento executado na inicialização do aplicativo"""
    logger.info("Aplicativo iniciando. Configurando diretórios...")
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Diretório data criado: {DATA_DIR}")
    
    for dir_path in [TEMP_DIR, RESULTS_DIR, CONVERTED_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Diretório criado: {dir_path}")
    
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
        logger.info(f"Serviço de limpeza automática iniciado (intervalo: {CLEANUP_INTERVAL_HOURS}h)")
        logger.info(f"Retenção: uploads: {TEMP_RETENTION_HOURS}h, resultados: {RESULTS_RETENTION_HOURS}h")
    
    if has_recovery_system:
        initialize_recovery_features()
        logger.info("Sistema de recuperação para valores NaN inicializado")

@app.post("/process", response_model=JobStatus, summary="Enviar e processar documento")
async def process_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    gemini_extractor: GeminiExtractor = Depends(get_gemini_extractor)
):
    try:
        job_id = os.path.basename(file.filename).split('.')[0]
        file_location = os.path.join(TEMP_DIR, f"{job_id}_{file.filename}")
        
        with open(file_location, "wb") as file_object:
            content = await file.read()
            file_object.write(content)
        
        logger.info(f"Arquivo salvo em: {file_location}")

        if has_recovery_system:
            try:                
                async def protected_process():
                    return await document_service.process_document(
                        file_location, file.filename, gemini_extractor
                    )
                
                job_id = await ProcessingRecovery.retry_processing_with_fixes(
                    protected_process, max_retries=3
                )
                
                if not job_id:
                    raise ValueError("Falha no processamento após múltiplas tentativas")
                
            except ImportError:
                # Fallback para versão sem proteção
                job_id = await document_service.process_document(
                    file_location, file.filename, gemini_extractor
                )
        else:
            # Processamento normal
            job_id = await document_service.process_document(
                file_location, file.filename, gemini_extractor
            )
        
        # Retornar o status inicial
        job = job_service.get_job(job_id)
        return JobStatus(**job)
    
    except Exception as e:
        logger.exception("Erro ao processar documento")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}/excel", summary="Obter resultado em Excel")
async def get_job_excel(job_id: str, season: str = None):
    """
    Retorna os resultados do job em formato Excel.
    
    - **job_id**: ID do job
    - **season**: Temporada (opcional, ex: "FW23")
    
    Retorna um arquivo Excel com os dados extraídos.
    """
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job ainda em processamento")
    
    # Verificar se temos resultados do Gemini
    if "gemini" not in job["model_results"] or "result" not in job["model_results"]["gemini"]:
        raise HTTPException(status_code=404, detail="Resultados não disponíveis")
    
    try:
        # Extrair dados do resultado
        extraction_result = job["model_results"]["gemini"]["result"]
        
        # Sanitizar valores NaN se tiver sistema de recuperação
        if has_recovery_system:
            try:
                from app.utils.recovery_system import ProcessingRecovery
                extraction_result = ProcessingRecovery.fix_extraction_result(extraction_result)
            except ImportError:
                pass
        
        # Criar DataFrame a partir dos dados
        df = create_dataframe_from_extraction(extraction_result, season)
        
        # Substituir valores NaN por 0 para garantir que o Excel funcione
        import math
        for col in df.columns:
            if df[col].dtype == 'float64' or df[col].dtype == 'float32':
                df[col] = df[col].fillna(0.0)
                # Substituir infinito por 0
                df[col] = df[col].replace([float('inf'), float('-inf')], 0.0)
        
        # Gerar ou recuperar o arquivo Excel
        excel_path = os.path.join(RESULTS_DIR, f"{job_id}_result.xlsx")
        
        # Verificar se já existe (para não reprocessar desnecessariamente)
        if not os.path.exists(excel_path):
            # Exportar para Excel
            df.to_excel(excel_path, index=False)
            logger.info(f"Arquivo Excel gerado: {excel_path}")
        
        # Retornar o arquivo
        return FileResponse(
            path=excel_path,
            filename=f"{job_id}_result.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        logger.exception(f"Erro ao gerar Excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar Excel: {str(e)}")

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    logger.info(f"Tentando acessar job com ID: {job_id}")
    
    # Listar todos os jobs disponíveis para debug
    all_jobs = job_service.list_jobs()
    logger.info(f"Jobs disponíveis: {list(all_jobs.keys())}")
    
    job = job_service.get_job(job_id)
    if not job:
        logger.error(f"Job não encontrado: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job não encontrado: {job_id}")
    
    logger.info(f"Job encontrado: {job_id}, status: {job['status']}")
    return JobStatus(**job)
    
@app.get("/job/{job_id}/json", summary="Obter resultado em JSON")
async def get_job_json(job_id: str):
    job = job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job ainda em processamento")
    
    # Verificar se temos resultados do Gemini
    if "gemini" not in job["model_results"] or "result" not in job["model_results"]["gemini"]:
        raise HTTPException(status_code=404, detail="Resultados não disponíveis")
    
    try:
        # Extrair dados do resultado
        extraction_result = job["model_results"]["gemini"]["result"]
        
        # Sanitizar valores NaN se tiver sistema de recuperação
        if has_recovery_system:
            try:
                from app.utils.recovery_system import ProcessingRecovery
                extraction_result = ProcessingRecovery.fix_extraction_result(extraction_result)
            except ImportError:
                # Sanitização básica para garantir JSON válido
                import math
                
                def sanitize_nan(obj):
                    if isinstance(obj, dict):
                        return {k: sanitize_nan(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [sanitize_nan(item) for item in obj]
                    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                        return None
                    else:
                        return obj
                
                extraction_result = sanitize_nan(extraction_result)
        
        return JSONResponse(content=extraction_result, status_code=200)
    except Exception as e:
        logger.exception(f"Erro ao gerar JSON: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar JSON: {str(e)}")

@app.get("/jobs", summary="Listar todos os jobs")
async def list_jobs():
    """
    Lista todos os jobs ativos no sistema.
    
    Retorna um dicionário com os IDs dos jobs e seus status.
    """
    return job_service.list_jobs()

@app.get("/", summary="Verificar status da API")
async def root():
    """
    Verifica se a API está funcionando.
    
    Retorna uma mensagem de status.
    """
    return {
        "message": "API de Extração de Documentos com Agentes Avançados está funcionando",
        "swagger_ui": "Acesse /docs para a interface interativa",
        "status": "online",
        "version": APP_VERSION
    }

def create_dataframe_from_extraction(extraction_result: Dict[str, Any], season: Optional[str] = None) -> pd.DataFrame:
    """
    Cria um DataFrame pandas a partir dos resultados da extração.
    
    Args:
        extraction_result: Resultados da extração
        season: Temporada (opcional)
        
    Returns:
        pd.DataFrame: DataFrame estruturado com os dados
    """
    # Dados para o DataFrame
    data = []
    
    # Dicionário para rastrear códigos de material e suas contagens
    material_code_counts = {}
    
    # Obter informações do pedido
    order_info = extraction_result.get("order_info", {})
    
    # Definir temporada, usando o parametro ou a informação do contexto
    current_season = season or order_info.get("season", "")
    
    # Processar cada produto
    for product in extraction_result.get("products", []):
        product_name = product.get("name", "")
        material_code_base = product.get("material_code", "")
        
        # Verificar se este código de material já foi processado antes
        if material_code_base in material_code_counts:
            material_code_counts[material_code_base] += 1
            # Adicionar sufixo ao código de material
            material_code = f"{material_code_base}.{material_code_counts[material_code_base]}"
        else:
            material_code_counts[material_code_base] = 1
            # Adicionar sufixo .1 ao primeiro produto com este código
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
                
                # Adicionar linha ao DataFrame
                data.append({
                    "Material Code": material_code,  # Código com sufixo numérico
                    "Base Code": material_code_base, # Opcional: manter o código base também
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
        
        # Ordenar colunas para melhor visualização
        column_order = [
            "Material Code", "Base Code", "Product Name", "Category", "Model",
            "Color Code", "Color Name", "Size", "Quantity",
            "Unit Price", "Sales Price", "Brand", "Supplier",
            "Season", "Order Number", "Date", "Document Type"
        ]
        
        # Filtrar apenas colunas existentes
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        return df
    else:
        # Retornar DataFrame vazio com as colunas necessárias
        return pd.DataFrame(columns=[
            "Material Code", "Base Code", "Product Name", "Category", "Model",
            "Color Code", "Color Name", "Size", "Quantity",
            "Unit Price", "Sales Price", "Brand", "Supplier",
            "Season", "Order Number", "Date", "Document Type"
        ])

# Entrada principal da aplicação
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("Iniciando servidor de extração de documentos com agentes avançados...")
    print("="*50)
    print(f"\nAcesse o Swagger UI: http://localhost:8000/docs")
    print(f"Diretório de uploads: {os.path.abspath(TEMP_DIR)}")
    print(f"Diretório de resultados: {os.path.abspath(RESULTS_DIR)}")
    print("\nPressione Ctrl+C para encerrar o servidor.")
    print("="*50 + "\n")
    
    # Iniciar o servidor
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)