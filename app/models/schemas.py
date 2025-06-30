# app/models/schemas.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, List
from datetime import datetime

class JobStatus(BaseModel):
    """Modelo para status de um job de processamento"""
    job_id: str
    status: str
    progress: float = 0.0
    file_path: str
    filename: str
    created_at: str
    model_results: Dict[str, Dict[str, Any]] = {}
        
    # Configuração para Pydantic V2
    model_config = ConfigDict(protected_namespaces=())

class ProductColor(BaseModel):
    """Modelo para cor de um produto"""
    color_code: str
    sizes: List[Dict[str, Any]]
    unit_price: float
    subtotal: float

class Product(BaseModel):
    """Modelo para um produto extraído"""
    name: str
    material_code: Optional[str] = None
    category: str
    model: Optional[str] = None
    colors: List[ProductColor]
    total_price: float

class OrderInfo(BaseModel):
    """Modelo para informações do pedido"""
    order_number: Optional[str] = None
    date: Optional[str] = None
    total_pieces: Optional[int] = None
    total_value: Optional[float] = None

class ExtractionResult(BaseModel):
    """Modelo para o resultado da extração"""
    products: List[Product]
    order_info: OrderInfo
    _metadata: Optional[Dict[str, Any]] = None