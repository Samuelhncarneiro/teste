# app/utils/json_utils.py
import json
import math
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def is_json_serializable(obj: Any) -> bool:
    """
    Verifica se um objeto é serializável para JSON
    """
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def sanitize_for_json(
    obj: Any, 
    default_number: float = 0.0,
    default_str: str = "",
    max_depth: int = 100,
    current_depth: int = 0
) -> Any:
    """
    Sanitiza recursivamente um objeto para garantir que seja serializável para JSON.
    Substitui valores problemáticos como NaN e Infinity por defaults seguros.
    """
    if current_depth > max_depth:
        logger.warning(f"Profundidade máxima de recursão atingida ({max_depth})")
        return None
    
    if obj is None:
        return None
    
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            logger.debug(f"Valor numérico inválido (NaN/Infinity) substituído por {default_number}")
            return default_number
        return obj
    
    if isinstance(obj, str):
        if not obj or not is_json_serializable(obj):
            return default_str
        return obj
    
    if isinstance(obj, dict):
        return {
            k: sanitize_for_json(
                v, 
                default_number=default_number,
                default_str=default_str,
                max_depth=max_depth,
                current_depth=current_depth + 1
            ) 
            for k, v in obj.items()
        }
    
    if isinstance(obj, (list, tuple)):
        return [
            sanitize_for_json(
                item, 
                default_number=default_number,
                default_str=default_str,
                max_depth=max_depth,
                current_depth=current_depth + 1
            ) 
            for item in obj
        ]
    
    if is_json_serializable(obj):
        return obj
    
    try:
        return str(obj)
    except:
        logger.warning(f"Objeto não serializável do tipo {type(obj)} substituído por None")
        return None

def safe_json_dump(obj: Any, file_path: str, **kwargs) -> bool:
    """
    Salva um objeto como JSON de forma segura, garantindo sanitização prévia
    """
    try:
        sanitized_obj = sanitize_for_json(obj)
        
        if 'indent' not in kwargs:
            kwargs['indent'] = 2
        if 'ensure_ascii' not in kwargs:
            kwargs['ensure_ascii'] = False
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sanitized_obj, f, **kwargs)
        
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar JSON: {str(e)}")
        
        try:
            logger.warning("Tentando recuperação com sanitização agressiva")
            sanitized_obj = sanitize_for_json(obj, default_number=0.0, default_str="")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_obj, f, **kwargs)
            
            return True
        except Exception as e2:
            logger.error(f"Falha na recuperação: {str(e2)}")
            return False

def fix_nan_in_products(products: List[Dict[str, Any]], markup: float = 2.73) -> List[Dict[str, Any]]:
    """
    Corrige valores NaN em produtos, recalculando preços quando necessário
    """
    fixed_products = []
    
    for product in products:
        if not isinstance(product, dict):
            continue
        
        if "colors" in product and isinstance(product["colors"], list):
            fixed_colors = []
            
            for color in product["colors"]:
                if "unit_price" not in color or color["unit_price"] is None or (
                    isinstance(color["unit_price"], float) and math.isnan(color["unit_price"])
                ):
                    color["unit_price"] = 0.0
                
                if "sales_price" not in color or color["sales_price"] is None or (
                    isinstance(color["sales_price"], float) and math.isnan(color["sales_price"])
                ):
                    color["sales_price"] = round(color["unit_price"] * markup, 2)
                
                if "subtotal" not in color or color["subtotal"] is None or (
                    isinstance(color["subtotal"], float) and math.isnan(color["subtotal"])
                ):
                    total_quantity = sum(
                        size.get("quantity", 0) 
                        for size in color.get("sizes", []) 
                        if size.get("quantity") is not None
                    )
                    color["subtotal"] = round(color["unit_price"] * total_quantity, 2)
                
                if "sizes" in color and isinstance(color["sizes"], list):
                    fixed_sizes = []
                    
                    for size in color["sizes"]:
                        if "quantity" not in size or size["quantity"] is None or (
                            isinstance(size["quantity"], float) and math.isnan(size["quantity"])
                        ):
                            size["quantity"] = 0
                        
                        if size.get("quantity", 0) > 0:
                            fixed_sizes.append(size)
                    
                    color["sizes"] = fixed_sizes
                
                if color.get("sizes", []):
                    fixed_colors.append(color)
            
            product["colors"] = fixed_colors
        
        if "total_price" not in product or product["total_price"] is None or (
            isinstance(product["total_price"], float) and math.isnan(product["total_price"])
        ):
            subtotals = [
                color.get("subtotal", 0) 
                for color in product.get("colors", []) 
                if color.get("subtotal") is not None
            ]
            product["total_price"] = sum(subtotals)
        
        if product.get("colors", []):
            fixed_products.append(product)
    
    return fixed_products